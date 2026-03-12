"""Application settings loaded from environment variables."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings

# Prevent OpenMP duplicate library crash (FAISS + numpy/torch on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class Settings(BaseSettings):
    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"

    # LLM
    llm_provider: str = "bedrock"  # "bedrock" | "openai" | "anthropic" | "azure" | "vertex"
    bedrock_model_id: str = "global.anthropic.claude-sonnet-4-6"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-10-21"

    # GCP Vertex AI
    gcp_project_id: str = ""
    gcp_location: str = "us-central1"
    vertex_model_id: str = "gemini-2.0-flash"

    # Embeddings
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # S3
    s3_bucket_name: str = "sales-deal-agent-docs"
    document_source: str = "local"  # "s3" | "local"

    # Search
    tavily_api_key: str = ""
    search_provider: str = "tavily"  # "tavily" | "duckduckgo"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "sales-deal-acceleration-agent"

    # Paths (relative to project root)
    crm_data_path: str = "data/crm/accounts.json"
    documents_path: str = "data/documents"
    faiss_index_path: str = "data/faiss_index"

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    model_config = {"env_file": ".env", "extra": "ignore"}


# Singleton instance
settings = Settings()
