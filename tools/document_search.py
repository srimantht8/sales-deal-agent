"""Document search tool — FAISS vectorstore with S3 or local document loading.

NOTE: FAISS is used here for demo/evaluation. Production deployment should use
Amazon OpenSearch Service for scalable, managed vector search.
"""

import logging

from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)

# Module-level singleton for the vectorstore
_vectorstore = None


class DocumentSearchInput(BaseModel):
    query: str = Field(description="The search query to find relevant internal documents")
    k: int = Field(default=4, description="Number of documents to return")


def load_documents_from_s3(bucket: str, prefix: str = "documents/") -> list[dict]:
    """Download documents from S3 bucket and return as list of {content, metadata}."""
    import boto3

    s3 = boto3.client(
        "s3",
        region_name=settings.aws_default_region,
    )

    documents = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".md"):
                continue
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            # Extract deal stage from path: documents/discovery/file.md → discovery
            parts = key.split("/")
            deal_stage = parts[1] if len(parts) >= 3 else "unknown"
            documents.append({
                "content": content,
                "metadata": {"source": key, "deal_stage": deal_stage},
            })
            logger.info(f"Loaded from S3: {key}")

    return documents


def load_documents_local(path: str | None = None) -> list[dict]:
    """Load markdown documents from local data/documents/ directory."""
    docs_path = settings.project_root / (path or settings.documents_path)
    documents = []

    for md_file in sorted(docs_path.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        # Extract deal stage from parent directory name
        deal_stage = md_file.parent.name
        documents.append({
            "content": content,
            "metadata": {"source": str(md_file.relative_to(docs_path)), "deal_stage": deal_stage},
        })
        logger.info(f"Loaded local: {md_file.name}")

    return documents


def build_vectorstore():
    """Build FAISS vectorstore from documents (S3 or local) and save to disk."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    from config.llm import get_embeddings

    # Load documents based on source setting
    source = settings.document_source.lower()
    if source == "s3":
        logger.info(f"Loading documents from S3: {settings.s3_bucket_name}")
        raw_docs = load_documents_from_s3(settings.s3_bucket_name)
    else:
        logger.info("Loading documents from local filesystem")
        raw_docs = load_documents_local()

    if not raw_docs:
        raise RuntimeError("No documents found. Check DOCUMENT_SOURCE and document paths.")

    # Convert to LangChain Document objects
    lc_docs = [
        Document(page_content=d["content"], metadata=d["metadata"])
        for d in raw_docs
    ]

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(lc_docs)
    logger.info(f"Split {len(lc_docs)} documents into {len(splits)} chunks")

    # Embed and create FAISS index
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Save to disk
    index_path = settings.project_root / settings.faiss_index_path
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info(f"FAISS index saved to {index_path}")

    return vectorstore


def load_vectorstore():
    """Load FAISS vectorstore from cache, or build if not found."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.vectorstores import FAISS

    from config.llm import get_embeddings

    index_path = settings.project_root / settings.faiss_index_path

    try:
        logger.info(f"Loading cached FAISS index from {index_path}")
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        logger.info("No cached FAISS index found, building from documents")
        _vectorstore = build_vectorstore()

    return _vectorstore


@tool("search_documents", args_schema=DocumentSearchInput)
def search_documents(query: str, k: int = 4) -> list[dict]:
    """Search internal company documents including case studies, technical guides,
    pricing information, proposals, and product roadmaps. Use specific keywords
    related to what the customer needs for best results."""
    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        logger.error(f"Failed to load vectorstore: {e}")
        return [{"error": str(e)}]

    results = vectorstore.similarity_search_with_score(query, k=k)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "deal_stage": doc.metadata.get("deal_stage", "unknown"),
            "similarity_score": round(float(score), 4),
        }
        for doc, score in results
    ]
