import re
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..cli_display import log
from .cancellation import check_cancelled


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


class LLMClient(ABC):

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0,
                 stream: bool = True, max_output_tokens: int = 16384):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stream = stream
        self.max_output_tokens = max_output_tokens
        self._stream_callback: Optional[Callable[[int], None]] = None

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
                        # Jittered exponential backoff
                        import random
                        wait = self.retry_delay * (2 ** (attempt - 1))
                        jitter = wait * 0.1 * random.random()
                        time.sleep(wait + jitter)
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
                    # Jittered exponential backoff
                    import random
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    jitter = wait * 0.1 * random.random()
                    
                    # Special handling for 429: wait longer
                    if "429" in str(e):
                        wait *= 2
                        log.info(f"[LLM] Rate limit detected (429). Backing off for {wait:.1f}s")
                    
                    time.sleep(wait + jitter)

        raise LLMError(
            f"LLM failed after {self.max_retries} retries: {last_error}")

    # ── Subclass hooks ──

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
