import re
from abc import ABC, abstractmethod
from ..llm.base import LLMClient


# ---------------------------------------------------------------------------
# Context deduplication
# ---------------------------------------------------------------------------

_PARA_SPLIT = re.compile(r'\n{2,}')
_WHITESPACE = re.compile(r'\s+')


def uniquify_context(text: str, min_size: int = 100) -> str:
    """Remove duplicate text blocks from assembled LLM context.

    Splits on blank-line boundaries (paragraph level), normalises each block
    (strip + collapse whitespace + lowercase) and drops any block whose
    normalised form has already been seen.

    Blocks shorter than *min_size* characters (after normalisation) are kept
    unconditionally so that small repeated markers, separators and headers
    are not accidentally removed.

    Parameters
    ----------
    text:
        The raw context string, possibly containing duplicated sections.
    min_size:
        Minimum normalised length for a block to be eligible for dedup.
        Blocks below this threshold are always kept.

    Returns
    -------
    str
        The context with duplicate blocks removed.
    """
    if not text:
        return text

    paragraphs = _PARA_SPLIT.split(text)
    seen: set[str] = set()
    kept: list[str] = []

    for para in paragraphs:
        normalized = _WHITESPACE.sub(' ', para.strip()).lower()

        # Always keep short blocks (headers, separators, labels)
        if len(normalized) < min_size:
            kept.append(para)
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        kept.append(para)

    return '\n\n'.join(kept)


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------

class Agent(ABC):
    def __init__(self, name: str, role: str, goal: str, llm_client: LLMClient,
                 prompt_suffix: str = ""):
        self.name = name
        self.role = role
        self.goal = goal
        self.llm_client = llm_client
        self.prompt_suffix = prompt_suffix

    @abstractmethod
    def process(self, task: str, context: str = "") -> str:
        """
        Process the given task and return the result.
        """
        pass

    def _build_prompt(self, task: str, context: str, language: str | None = None) -> str:
        import os as _os
        prompt = f"Role: {self.role}\nGoal: {self.goal}\n\n"
        if language:
            from ..language import get_language_name
            prompt += f"Language: {get_language_name(language)}\n\n"
        if _os.name == 'nt':
            prompt += "Platform: Windows (use cmd.exe-compatible commands)\n\n"
        else:
            import platform as _platform
            _sysname = _platform.system()
            _os_label = "macOS" if _sysname == "Darwin" else _sysname
            prompt += f"Platform: {_os_label}\n\n"
        if self.prompt_suffix:
            prompt += f"Instructions: {self.prompt_suffix}\n\n"
        if context:
            context = uniquify_context(context)
            prompt += f"Context: {context}\n\n"
        prompt += f"Task: {task}\n\nResponse:"
        return prompt
