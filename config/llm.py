"""LLM and embeddings factory functions."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from config.settings import settings

logger = logging.getLogger(__name__)

# Cache LLM instances by (provider, temperature) to avoid re-creating clients
_llm_cache: dict[tuple[str, float], BaseChatModel] = {}
_embeddings_cache: Embeddings | None = None


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return the configured LLM based on LLM_PROVIDER setting. Cached by (provider, temperature)."""
    provider = settings.llm_provider.lower()
    cache_key = (provider, temperature)

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if provider == "bedrock":
        import botocore.config
        from langchain_aws import ChatBedrockConverse

        llm = ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_default_region,
            temperature=temperature,
            config=botocore.config.Config(read_timeout=300),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            temperature=temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=settings.anthropic_api_key,
            temperature=temperature,
        )
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            api_key=settings.azure_openai_api_key,
            temperature=temperature,
        )
    elif provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(
            model=settings.vertex_model_id,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. "
            "Use 'bedrock', 'openai', 'anthropic', 'azure', or 'vertex'."
        )

    _llm_cache[cache_key] = llm
    return llm


def clear_llm_cache():
    """Clear the LLM instance cache. Used by the provider comparison benchmark."""
    _llm_cache.clear()


def get_embeddings() -> Embeddings:
    """Return the configured embeddings model. Cached after first call.

    Priority: Bedrock Titan → OpenAI → HuggingFace (local, no API key needed).
    """
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    # Try Bedrock if AWS credentials are available
    if settings.aws_access_key_id:
        try:
            from langchain_aws import BedrockEmbeddings

            _embeddings_cache = BedrockEmbeddings(
                model_id=settings.embedding_model_id,
                region_name=settings.aws_default_region,
            )
            return _embeddings_cache
        except Exception as e:
            logger.warning(f"Bedrock embeddings failed ({e}), trying fallbacks")

    # Try OpenAI
    if settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        _embeddings_cache = OpenAIEmbeddings(api_key=settings.openai_api_key)
        return _embeddings_cache

    # Fall back to local HuggingFace embeddings (no API key needed)
    logger.info("Using local HuggingFace embeddings (no cloud API keys configured)")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    _embeddings_cache = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_cache
