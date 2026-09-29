from dataclasses import dataclass, field
from typing import Optional


class ModelProvider:
    OLLAMA = "ollama"
    # llmman serves the Ollama API on port 17434 (https://github.com/llmmanorg/llmman)
    LLMMAN = "llmman"
    HUGGINGFACE = "huggingface"

    OLLAMA_COMPATIBLE = (OLLAMA, LLMMAN)


def _default_none() -> None:
    return None


@dataclass
class QueryPipelineConfig:
    """Base configuration for all pipelines."""

    # Required parameters
    es_url: str

    # Common provider parameters
    provider: str = field(default=ModelProvider.OLLAMA)
    ollama_url: Optional[str] = None
    hf_api_key: Optional[str] = field(default_factory=_default_none)

    embedding_model: Optional[str] = None
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    allow_model_pull: bool = field(default=True)

    # Elasticsearch parameters
    es_index: Optional[str] = None
    es_top_k: Optional[int] = None
    es_num_candidates: Optional[int] = None
    es_basic_auth_user: Optional[str] = None
    es_basic_auth_password: Optional[str] = None

    # Conversation analysis and logging
    enable_conversation_logs: Optional[bool] = True

    # Core generation parameters
    context_window: Optional[int] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    top_k: Optional[int] = None

    # Advanced sampling parameters
    top_p: Optional[float] = None
    min_p: Optional[float] = None

    # Mirostat parameters
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None

    # Repetition control
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None

    # Generation control
    num_predict: Optional[int] = None
    tfs_z: Optional[float] = None
    stop_sequence: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.provider == ModelProvider.HUGGINGFACE and not self.hf_api_key:
            raise ValueError(
                "HuggingFace API key is required when using HuggingFace provider"
            )

        if self.provider not in [
            *ModelProvider.OLLAMA_COMPATIBLE,
            ModelProvider.HUGGINGFACE,
        ]:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if self.provider in ModelProvider.OLLAMA_COMPATIBLE and not self.ollama_url:
            raise ValueError(f"ollama_url is required when using {self.provider}")
