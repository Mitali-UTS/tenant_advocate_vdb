"""
FastAPI application for the ChromaDB vector search API.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from vector_store import ChromaVectorStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (all values come from environment variables so secrets stay out of
# source code; sensible defaults are provided for local development)
# ---------------------------------------------------------------------------

COLLECTION_NAME     = "tenancy_documents"
PERSIST_DIRECTORY   = "vdb/tenancy_regulations"
EMBEDDING_MODEL     = "microsoft/harrier-oss-v1-0.6b"
DEFAULT_TOP_K       = 5
API_KEY             = ""

# ---------------------------------------------------------------------------
# Optional API-key authentication
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: Optional[str] = Security(api_key_header)) -> None:
    """If API_KEY env var is set, every request must supply it."""
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Application lifespan – initialise the store once at startup
# ---------------------------------------------------------------------------

store: ChromaVectorStore  # module-level reference populated at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    logger.info("Initialising ChromaVectorStore …")
    store = ChromaVectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_model_name=EMBEDDING_MODEL,
    )
    logger.info("Store ready: %s", store.health())
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ChromaDB Vector Search API",
    description=(
        "Semantic search over a ChromaDB collection using "
        "sentence-transformer embeddings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=100, description="Number of results to return.")


class SimpleFilteredQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=100, description="Number of results to return.")
    filters: dict[str, str | int | float | bool] = Field(
        ...,
        description=(
            "Simple key=value equality filters applied to document metadata. "
            'Example: {"source": "arxiv"} or {"source": "arxiv", "year": 2023}'
        ),
    )


class FilteredQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=100, description="Number of results to return.")
    where: dict[str, Any] = Field(
        ...,
        description=(
            "ChromaDB metadata filter. "
            'Examples: {"source": "arxiv"} | {"year": {"$gte": 2022}} | '
            '{"$and": [{"source": "arxiv"}, {"year": {"$gte": 2022}}]}'
        ),
    )


class SearchResult(BaseModel):
    id: str
    document: Optional[str]
    metadata: dict[str, Any]
    distance: Optional[float]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


class ReloadRequest(BaseModel):
    embedding_model_name: Optional[str] = Field(
        None, description="New model name to hot-swap the embedding model."
    )
    persist_directory: Optional[str] = Field(
        None, description="New persist path (re-connects the ChromaDB client)."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health_check():
    """Returns basic service health and collection stats."""
    return {"status": "ok", **store.health()}


@app.post(
    "/query",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic search (no filters)",
)
def query(
    body: QueryRequest,
    _: None = Depends(verify_api_key),
) -> SearchResponse:
    """
    Embed the query and return the top-k most similar documents.
    """
    try:
        results = store.query(query_text=body.query, top_k=body.top_k)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(
        query=body.query,
        results=[SearchResult(**r) for r in results],
        total=len(results),
    )


@app.post(
    "/query/filtered",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic search with metadata filters",
)
def query_filtered(
    body: FilteredQueryRequest,
    _: None = Depends(verify_api_key),
) -> SearchResponse:
    """
    Embed the query and return the top-k most similar documents that
    also satisfy the supplied ChromaDB `where` filter.

    Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`,
    `$in`, `$nin`, `$and`, `$or`.
    """
    try:
        results = store.query_with_filters(
            query_text=body.query,
            where=body.where,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("Filtered query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(
        query=body.query,
        results=[SearchResult(**r) for r in results],
        total=len(results),
    )


@app.post(
    "/query/where",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic search with simple key=value metadata filters",
)
def query_where(
    body: SimpleFilteredQueryRequest,
    _: None = Depends(verify_api_key),
) -> SearchResponse:
    """
    The simplest filtered search — pass a plain dict of metadata
    key=value pairs and the API handles ChromaDB filter syntax for you.

    Single filter:   `{"source": "arxiv"}`
    Multiple fields: `{"source": "arxiv", "year": 2023}`  ← treated as AND
    """
    try:
        results = store.query_where(
            query_text=body.query,
            filters=body.filters,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("where-filter query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(
        query=body.query,
        results=[SearchResult(**r) for r in results],
        total=len(results),
    )


@app.post(
    "/reload",
    tags=["ops"],
    summary="Hot-reload the embedding model or reconnect to a new persist path",
)
def reload(
    body: ReloadRequest,
    _: None = Depends(verify_api_key),
):
    """
    Hot-reload the embedding model and/or ChromaDB connection without
    restarting the server process.  Useful after updating the model or
    swapping the data directory on Render.
    """
    try:
        store.reload(
            embedding_model_name=body.embedding_model_name,
            persist_directory=body.persist_directory,
        )
    except Exception as exc:
        logger.exception("Reload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "reloaded", **store.health()}
