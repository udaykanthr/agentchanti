from .base import LLMClient, LLMError, ToolsNotSupportedError
from .chat_types import ChatResponse, Message, ToolCall, ToolDef, flatten_messages
from .ollama import OllamaClient
from .lm_studio import LMStudioClient
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .anthropic_client import AnthropicClient


def build_embed_client(cfg, **llm_kwargs) -> "LLMClient | None":
    """
    Build an LLM client for embeddings, using ``cfg.EMBEDDING_PROVIDER`` when
    set (falling back to ``cfg.PROVIDER``).

    Returns ``None`` when the resolved provider doesn't support embeddings
    (i.e. Anthropic), so callers can disable the embedding step gracefully.
    """
    embed_provider = cfg.EMBEDDING_PROVIDER or cfg.PROVIDER
    embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL

    if embed_provider == "anthropic":
        return None  # Anthropic has no embedding API

    if embed_provider == "ollama":
        return OllamaClient(
            base_url=cfg.OLLAMA_BASE_URL, model=embed_model, **llm_kwargs)

    if embed_provider == "openai":
        return OpenAIClient(
            base_url=cfg.OPENAI_BASE_URL, model=embed_model,
            api_key=cfg.OPENAI_API_KEY, **llm_kwargs)

    if embed_provider == "gemini":
        return GeminiClient(
            base_url=cfg.GEMINI_BASE_URL, model=embed_model,
            api_key=cfg.GEMINI_API_KEY, **llm_kwargs)

    # lm_studio / default
    return LMStudioClient(
        base_url=cfg.LM_STUDIO_BASE_URL, model=embed_model, **llm_kwargs)
