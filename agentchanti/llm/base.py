import re
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..cli_display import log
from .cancellation import check_cancelled
from .chat_types import ChatResponse, Message, ToolDef, flatten_messages


# Matches well-formed <think>...</think> blocks (including newlines).
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

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0,
                 stream: bool = True, max_output_tokens: int = 16384):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stream = stream
        self.max_output_tokens = max_output_tokens
        self._stream_callback: Optional[Callable[[int], None]] = None
        # Flipped off when the active model rejects native tool calling,
        # so we don't re-attempt (and re-fail) on every subsequent call.
        self._native_tools_ok = True

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
                if use_stream:
                    result = self._generate_stream(active_prompt)
                else:
                    result = self._generate(active_prompt)

                result = _strip_reasoning(result) if result else result

                if not result or not result.strip():
                    log.warning(
                        f"[LLM] Empty response on attempt {attempt}/{self.max_retries}")
                    if attempt < self.max_retries:
                        # Prefix the prompt with an explicit instruction to
                        # suppress reasoning-only output.  Some models (e.g.
                        # deepseek-r1 variants) emit all tokens inside <think>
                        # blocks that get stripped, leaving an empty response.
                        # Telling the model to skip the thinking step on retry
                        # is the most reliable way to get visible output.
                        active_prompt = (
                            "[IMPORTANT: Your previous response was empty. "
                            "Do NOT use <think> tags, reasoning blocks, or any "
                            "XML-style wrapper tags. Output your answer directly "
                            "with no preamble.]\n\n"
                            + prompt
                        )
                        self._backoff(attempt)
                        continue
                    raise LLMError("LLM returned empty response after all retries")

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
            f"LLM failed after {self.max_retries} retries: {last_error}")

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
