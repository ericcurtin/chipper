import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.pipeline_config import ModelProvider, QueryPipelineConfig  # noqa: E402


@pytest.mark.parametrize("provider", ModelProvider.OLLAMA_COMPATIBLE)
def test_ollama_compatible_requires_url(provider):
    with pytest.raises(ValueError, match="ollama_url is required"):
        QueryPipelineConfig(es_url="http://es", provider=provider)


@pytest.mark.parametrize("provider", ModelProvider.OLLAMA_COMPATIBLE)
def test_ollama_compatible_accepts_url(provider):
    config = QueryPipelineConfig(
        es_url="http://es", provider=provider, ollama_url="http://localhost:17434"
    )
    assert config.ollama_url == "http://localhost:17434"
