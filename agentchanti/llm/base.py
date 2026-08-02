import re
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..cli_display import log
from .cancellation import check_cancelled
from .chat_types import ChatResponse, Message, ToolDef, flatten_messages


# Matches well-formed <think>...</think> blocks (including newlines).
_TRUSTSTORE_APPLIED = False


def ensure_system_trust_store() -> None:
    """Route TLS verification through the OS trust store, once per process.

    certifi's bundle does not contain the issuing CA on machines where a
    corporate proxy or endpoint-security product terminates TLS, so every
    provider call dies with:

        SSLError: certificate verify failed: unable to get local issuer
        certificate

    — at the first LLM request, with nothing pointing at the cause. The
    `truststore` package resolves it by validating against the platform
    store, which does have the injected root. Idempotent, best-effort, and
    a no-op when truststore is absent or already active.
    """
    global _TRUSTSTORE_APPLIED
    if _TRUSTSTORE_APPLIED:
        return
    _TRUSTSTORE_APPLIED = True
    try:
        import truststore
    except ImportError:
        return
    try:
        truststore.inject_into_ssl()
        log.debug("[LLM] TLS verification routed through the OS trust store")
    except Exception as exc:      # never block a run on this
        log.debug("[LLM] truststore injection skipped: %s", exc)


def explain_tls_failure(exc: Exception) -> str:
    """Extra guidance appended to a certificate-verification error."""
    text = str(exc)
    if "certificate verify failed" not in text.lower():
        return ""
    try:
        import truststore     # noqa: F401
        hint = ("truststore is installed but did not resolve it — check "
                "whether the intercepting CA is in the OS trust store")
    except ImportError:
        hint = "install it with `pip install truststore` and re-run"
    return (
        "\n\nTLS certificate verification failed. This usually means a "
        "proxy or endpoint-security product is terminating TLS and its CA "
        "is not in certifi's bundle; " + hint + "."
    )


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
# Matches a leading reasoning block where the opening <think> was lost
# (e.g. truncated by streaming) but the closing tag survived.
_DANGLING_CLOSE_RE = re.compile(r"\A.*?</think>", re.DOTALL | re.IGNORECASE)
# Matches an unterminated <think> at the very start with no closer anywhere.
_DANGLING_OPEN_RE = re.compile(r"\A\s*<think\b[^>]*>", re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by reasoning models.

    Handles three cases:
      1. Well-formed paired tags anywhere in the response.
      2. A dangling </think> at/near the start (opener lost to truncation).
      3. An unterminated <think> at the start with no closer (drop the rest
         of the response is too aggressive — instead just drop the opener
         and let downstream parsers see the raw text).

    Non-reasoning models are unaffected: if no <think> tag is present, the
    input is returned unchanged.
    """
    if not text:
        return text
    lowered = text.lower()
    if "<think" not in lowered and "</think>" not in lowered:
        return text

    cleaned = _THINK_BLOCK_RE.sub("", text)

    # If a stray </think> still appears (opener was truncated), drop everything
    # from the start of the response up to and including that closer.
    if "</think>" in cleaned.lower():
        cleaned = _DANGLING_CLOSE_RE.sub("", cleaned, count=1)

    # Drop a stray opener with no closer.
    cleaned = _DANGLING_OPEN_RE.sub("", cleaned, count=1)

    return cleaned.lstrip()


class LLMError(Exception):
    """Raised when all LLM retries are exhausted."""


class NonRetryableLLMError(LLMError):
    """A request the provider will reject identically every time.

    A malformed request (4xx other than 408/409/429) does not become valid
    by being sent again. Retrying it three times with exponential backoff
    only delays the inevitable failure and spends quota doing it.
    """


class ToolsNotSupportedError(Exception):
    """Raised by a provider when the active model rejects native tool calling.

    ``LLMClient.chat()`` catches this, disables native tools for the rest of
    the session and retries the call through the text-flattening fallback.
    """


class LLMClient(ABC):

    #: Providers with a native multi-turn chat endpoint (messages + tools)
    #: set this to True and implement ``_chat``. Others get the
    #: text-flattening fallback automatically.
    NATIVE_CHAT = False

    #: Provider stop-reason strings meaning "stopped at the output-token
    #: cap" (OpenAI ``length``, Anthropic ``max_tokens``, Ollama ``length``).
    _TOKEN_LIMIT_REASONS = ("length", "max_tokens", "max_output_tokens")

    # Below this share of the output budget, VISIBLE output is a sliver and
    # a cap hit means the budget was spent on hidden reasoning rather than
    # on an answer that was genuinely too long. Deliberately low: a real
    # long answer cut short sits near 100%, a burn near zero, so anything
    # in between stays on the truncation path and is returned to the caller.
    _HIDDEN_BURN_MAX_VISIBLE_SHARE = 0.25

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0,
                 stream: bool = True, max_output_tokens: int = 16384):
        # Before any provider call: on a TLS-intercepted machine certifi
        # cannot build a chain and every request fails with an opaque
        # SSLError.
        ensure_system_trust_store()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stream = stream
        self.max_output_tokens = max_output_tokens
        self._stream_callback: Optional[Callable[[int], None]] = None
        # Flipped off when the active model rejects native tool calling,
        # so we don't re-attempt (and re-fail) on every subsequent call.
        self._native_tools_ok = True
        # Side-channel for the text path: providers set this to their raw
        # stop reason on every ``_generate``/``_generate_stream`` so
        # ``generate_response`` can tell an empty/short answer that hit the
        # output-token cap (reasoning burn / truncation) from a clean stop.
        self._last_stop_reason: str = ""
        # True when the last ``generate_response`` returned NON-empty text
        # that was cut at the token cap — a truncated result the caller
        # (e.g. the planner) should treat as incomplete.
        self._last_truncated: bool = False
        # VISIBLE completion tokens of the last call, as the provider
        # reported them. Hidden reasoning/thinking is charged to the same
        # output budget but excluded here, which is what makes a burn
        # distinguishable from an answer that was genuinely too long.
        # 0 means "not reported" and disables the burn check.
        self._last_completion_tokens: int = 0

    def set_stream_callback(self, callback: Callable[[int], None]) -> None:
        """Set a callback that receives ``(tokens_generated)`` during streaming."""
        self._stream_callback = callback

    # ── Public entry point ──

    def generate_response(self, prompt: str) -> str:
        """Generate a response with automatic retry and exponential backoff.

        Calls ``_generate_stream`` when streaming is enabled, otherwise
        ``_generate``.  Raises :class:`LLMError` after all retries are
        exhausted.
        """
        last_error: Exception | None = None
        use_stream = self.stream  # mutable — falls back on failure
        # Prompt sent to the model — may be prefixed on empty-response retries
        # to suppress reasoning-only output (e.g. <think> tags that get stripped).
        active_prompt = prompt

        for attempt in range(1, self.max_retries + 1):
            try:
                self._last_stop_reason = ""
                if use_stream:
                    result = self._generate_stream(active_prompt)
                else:
                    result = self._generate(active_prompt)

                result = _strip_reasoning(result) if result else result
                hit_cap = self._generate_hit_token_limit()

                if not result or not result.strip():
                    if attempt < self.max_retries:
                        if hit_cap:
                            # The whole output budget was spent with nothing
                            # visible — a reasoning model burned every token
                            # thinking (observed: minimax planner, 16384
                            # tokens, empty text). A verbatim retry is a coin
                            # flip; let the provider dial reasoning down.
                            log.warning(
                                f"[LLM] Empty response at the output-token "
                                f"limit on attempt {attempt}/{self.max_retries}"
                                f" — reasoning burn; requesting reduced effort")
                            self._prepare_token_limit_retry()
                        else:
                            log.warning(
                                f"[LLM] Empty response on attempt "
                                f"{attempt}/{self.max_retries}")
                            # Some models (e.g. deepseek-r1 variants) emit all
                            # tokens inside <think> blocks that get stripped,
                            # leaving an empty response. Telling the model to
                            # skip the thinking step is the most reliable way
                            # to get visible output.
                            active_prompt = (
                                "[IMPORTANT: Your previous response was empty. "
                                "Do NOT use <think> tags, reasoning blocks, or "
                                "any XML-style wrapper tags. Output your answer "
                                "directly with no preamble.]\n\n"
                                + prompt
                            )
                        self._backoff(attempt)
                        continue
                    log.warning(
                        f"[LLM] Empty response on attempt "
                        f"{attempt}/{self.max_retries}")
                    raise LLMError("LLM returned empty response after all retries")

                # A cap hit with only a sliver of VISIBLE output means the
                # budget went somewhere invisible — hidden thinking. That
                # is the same burn handled above for empty responses, and
                # it was falling through here simply because a few tokens
                # of prose escaped. Observed on Gemini: 654 visible tokens
                # of a 16,384 budget, cut mid-file, twice with identical
                # counts — a verbatim retry reproduces it exactly, so the
                # step failed having written nothing.
                if (hit_cap and attempt < self.max_retries
                        and self._looks_like_hidden_burn()):
                    log.warning(
                        "[LLM] Only %d visible tokens of a %d budget before "
                        "the cap on attempt %d/%d — the rest went to hidden "
                        "reasoning; retrying with it dialled down",
                        self._last_completion_tokens, self.max_output_tokens,
                        attempt, self.max_retries)
                    self._prepare_token_limit_retry()
                    self._backoff(attempt)
                    continue

                # Non-empty. Flag truncation (cut at the token cap) so callers
                # that need a complete answer — the planner — can detect a
                # partial result instead of running with a silent stub.
                self._last_truncated = hit_cap
                if hit_cap:
                    log.warning(
                        "[LLM] Response hit the output-token limit — result "
                        "is likely truncated (%d tokens)",
                        self.max_output_tokens)
                return result

            except LLMError:
                raise
            except Exception as e:
                # If Ctrl+C closed the response socket mid-stream the
                # underlying ConnectionError surfaces here — don't retry,
                # propagate as KeyboardInterrupt instead.
                check_cancelled()
                last_error = e
                log.warning(
                    f"[LLM] Error on attempt {attempt}/{self.max_retries}: {e}")
                
                # If streaming failed, fall back to non-streaming for next retry
                if use_stream:
                    log.warning("[LLM] Streaming failed — falling back to non-streaming")
                    use_stream = False
                
                if attempt < self.max_retries:
                    self._backoff(attempt, error=e)

        raise LLMError(
            f"LLM failed after {self.max_retries} retries: {last_error}"
            + explain_tls_failure(last_error))

    def chat(self, messages: List[Message],
             tools: Optional[List[ToolDef]] = None) -> ChatResponse:
        """Multi-turn chat with optional native tool calling.

        Providers that set ``NATIVE_CHAT = True`` and implement ``_chat``
        get retry with exponential backoff (mirroring ``generate_response``).
        All other providers — and models that reject tools at runtime — fall
        back to flattening the conversation into a single text prompt sent
        through ``generate_response`` (which carries its own retry).

        Raises :class:`LLMError` after all retries are exhausted.
        """
        if not self.NATIVE_CHAT:
            return self._chat_via_text(messages, tools)

        use_tools = tools if self._native_tools_ok else None
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._chat(messages, use_tools)
                if result.text:
                    result.text = _strip_reasoning(result.text)

                if result.is_empty:
                    if self._hit_token_limit(result):
                        # The entire output budget was consumed with
                        # nothing visible — reasoning models can burn
                        # every completion token "thinking" (observed:
                        # 16384 tokens, ~110s, empty text, zero tool
                        # calls). A verbatim retry is a coin flip; let
                        # the provider dial reasoning down first.
                        log.warning(
                            f"[LLM] Chat hit the output-token limit "
                            f"({self.max_output_tokens}) with no visible "
                            f"output on attempt {attempt}/"
                            f"{self.max_retries} — reasoning burn; "
                            f"requesting reduced effort for the retry")
                        self._prepare_token_limit_retry()
                    else:
                        log.warning(
                            f"[LLM] Empty chat response on attempt "
                            f"{attempt}/{self.max_retries}")
                    if attempt < self.max_retries:
                        self._backoff(attempt)
                        continue
                    raise LLMError(
                        "LLM returned empty chat response after all retries")

                return result

            except LLMError:
                raise
            except ToolsNotSupportedError as e:
                log.warning(
                    f"[LLM] Model does not support native tool calling — "
                    f"falling back to text mode: {e}")
                self._native_tools_ok = False
                return self._chat_via_text(messages, tools)
            except Exception as e:
                check_cancelled()
                last_error = e
                log.warning(
                    f"[LLM] Chat error on attempt "
                    f"{attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    self._backoff(attempt, error=e)

        raise LLMError(
            f"LLM chat failed after {self.max_retries} retries: {last_error}")

    @classmethod
    def _hit_token_limit(cls, result: ChatResponse) -> bool:
        """True when the provider stopped the response at the output-token
        cap (OpenAI ``length``, Anthropic ``max_tokens``, Ollama
        ``length``)."""
        return (result.stop_reason or "").lower() in cls._TOKEN_LIMIT_REASONS

    def _looks_like_hidden_burn(self) -> bool:
        """True when a cap hit produced only a sliver of visible output.

        The remaining budget went to hidden reasoning, so retrying the
        same prompt reproduces it exactly — the fix is to dial reasoning
        down, which is what the empty-response path already does. Returns
        False when the provider does not report completion tokens, so an
        unknown count never triggers a retry.
        """
        visible = self._last_completion_tokens
        if not visible or self.max_output_tokens <= 0:
            return False
        return (visible / self.max_output_tokens
                ) < self._HIDDEN_BURN_MAX_VISIBLE_SHARE

    def _generate_hit_token_limit(self) -> bool:
        """Text-path counterpart of :meth:`_hit_token_limit`, reading the
        stop reason the provider stashed during the last generate call."""
        return (self._last_stop_reason or "").lower() in self._TOKEN_LIMIT_REASONS

    def _prepare_token_limit_retry(self) -> None:
        """Hook invoked before retrying a chat whose response consumed the
        entire output budget with no visible text or tool calls (reasoning
        burn). Providers that can lower reasoning effort for the next
        request override this; the default retries unchanged."""

    def _backoff(self, attempt: int, error: Exception | None = None) -> None:
        """Sleep with jittered exponential backoff before a retry."""
        import random
        wait = self.retry_delay * (2 ** (attempt - 1))
        if error is not None and "429" in str(error):
            wait *= 2
            log.info(f"[LLM] Rate limit detected (429). Backing off for {wait:.1f}s")
        time.sleep(wait + wait * 0.1 * random.random())

    def _chat_via_text(self, messages: List[Message],
                       tools: Optional[List[ToolDef]] = None) -> ChatResponse:
        """Fallback: flatten the conversation into one text prompt.

        Tool calls cannot be returned on this path — callers should check
        ``supports_tools()`` and use a prompt-based protocol instead when
        it is False.
        """
        if tools:
            log.warning(
                "[LLM] Native tool calling unavailable for this provider/"
                "model — sending flattened text prompt; no structured tool "
                "calls will be returned")
        text = self.generate_response(flatten_messages(messages))
        return ChatResponse(text=text, stop_reason="stop")

    def supports_tools(self) -> bool:
        """True when native, schema-validated tool calling is available."""
        return self.NATIVE_CHAT and self._native_tools_ok

    # ── Subclass hooks ──

    def _chat(self, messages: List[Message],
              tools: Optional[List[ToolDef]] = None) -> ChatResponse:
        """Native chat implementation. Only called when ``NATIVE_CHAT``.

        Should raise :class:`ToolsNotSupportedError` when the provider
        reports that the active model cannot do tool calling.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement native chat")

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Synchronous (non-streaming) generation."""

    @abstractmethod
    def _generate_stream(self, prompt: str) -> str:
        """Streaming generation. Should call ``self._stream_callback``
        periodically with the number of tokens generated so far."""

    @abstractmethod
    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        """Generate an embedding vector for the given text."""

    def generate_embeddings_batch(self, texts: list[str], model: Optional[str] = None, **kwargs) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in a single API call.

        Subclasses should override this to use native batch endpoints.
        Default implementation falls back to sequential single calls.
        """
        return [self.generate_embedding(t, model=model, **kwargs) for t in texts]
