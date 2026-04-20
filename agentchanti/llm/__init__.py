from .base import LLMClient, LLMError
from .ollama import OllamaClient
from .lm_studio import LMStudioClient
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .anthropic_client import AnthropicClient


class MissingAPIKeyError(LLMError):
    """Raised when the chosen provider needs an API key the caller didn't set.

    Carries ``provider`` and ``env_var`` so callers can render a tailored
    error message without re-parsing the exception text.
    """

    def __init__(self, provider: str, env_var: str):
        super().__init__(f"{provider} provider requires {env_var}")
        self.provider = provider
        self.env_var = env_var


def build_llm_client(
    cfg,
    *,
    provider: str | None = None,
    model: str | None = None,
    **llm_kwargs,
) -> "LLMClient":
    """Build the main generation LLM client from config.

    Central factory so multiple entry points (main CLI, library api,
    testing module) don't each re-implement the provider switch. Resolves
    provider and model via the same priority as the rest of the codebase:
    explicit args > ``cfg`` > defaults.

    Raises ``MissingAPIKeyError`` when the resolved provider needs an API
    key that isn't set in ``cfg`` — callers decide how to surface it
    (print + exit on the CLI, re-raise in the library).
    """
    provider = provider or cfg.PROVIDER
    model = model or cfg.DEFAULT_MODEL

    if provider == "ollama":
        return OllamaClient(
            base_url=cfg.OLLAMA_BASE_URL, model=model, **llm_kwargs)

    if provider == "openai":
        if not cfg.OPENAI_API_KEY:
            raise MissingAPIKeyError("openai", "OPENAI_API_KEY")
        return OpenAIClient(
            base_url=cfg.OPENAI_BASE_URL, model=model,
            api_key=cfg.OPENAI_API_KEY, **llm_kwargs)

    if provider == "gemini":
        if not cfg.GEMINI_API_KEY:
            raise MissingAPIKeyError("gemini", "GEMINI_API_KEY")
        return GeminiClient(
            base_url=cfg.GEMINI_BASE_URL, model=model,
            api_key=cfg.GEMINI_API_KEY, **llm_kwargs)

    if provider == "anthropic":
        if not cfg.ANTHROPIC_API_KEY:
            raise MissingAPIKeyError("anthropic", "ANTHROPIC_API_KEY")
        return AnthropicClient(
            base_url=cfg.ANTHROPIC_BASE_URL, model=model,
            api_key=cfg.ANTHROPIC_API_KEY, **llm_kwargs)

    # lm_studio / default
    return LMStudioClient(
        base_url=cfg.LM_STUDIO_BASE_URL, model=model,
        reasoning_effort=cfg.LM_STUDIO_REASONING_EFFORT, **llm_kwargs)


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
