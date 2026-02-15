import os

class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-coder-v2-lite-instruct")
    Context_Window = 1024
