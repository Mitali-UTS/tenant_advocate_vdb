"""
ChromaDB vector store with sentence-transformer embeddings.
Designed for deployment on Render with a persistent disk.
"""

import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Wraps a ChromaDB persistent client and a sentence-transformer
    embedding model.  One instance per collection is the intended
    usage pattern; create multiple instances for multiple collections.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        distance_metric: str = "cosine",   # "cosine" | "l2" | "ip"
    ) -> None:
        """
        Parameters
        ----------
        collection_name     : Name of the ChromaDB collection to use.
        persist_directory   : Path on disk where ChromaDB persists data.
                              On Render, point this at your mounted disk,
                              e.g. /var/data/chroma.
        embedding_model_name: Any sentence-transformers model hub name.
        distance_metric     : Distance function used when the collection
                              is created for the first time.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.distance_metric = distance_metric

        self._client: chromadb.ClientAPI = None          # type: ignore[assignment]
        self._collection: chromadb.Collection = None     # type: ignore[assignment]
        self._model: SentenceTransformer = None          # type: ignore[assignment]

        self._initialize()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Load the embedding model and connect to the ChromaDB collection."""

        # Then inside _initialize in your class, add temporarily:
        import os
        print("Module cwd:", os.getcwd())
        print("Resolved path:", os.path.abspath(self.persist_directory))

        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self._model = SentenceTransformer(self.embedding_model_name, model_kwargs={"dtype": "auto"})

        logger.info("Connecting to ChromaDB at: %s", self.persist_directory)
        self._client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        logger.info("Opening / creating collection: %s", self.collection_name)
        self._collection = self._client.get_collection(
            name=self.collection_name
        )
        logger.info(
            "Ready. Collection '%s' contains %d documents.",
            self.collection_name,
            self._collection.count(),
        )

    def reload(
        self,
        embedding_model_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        """
        Hot-reload the embedding model and/or re-connect to a different
        persist directory without restarting the process.
        """
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name
        if persist_directory:
            self.persist_directory = persist_directory
        self._initialize()

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for *text*."""
        return self._model.encode(text, normalize_embeddings=True).tolist()

    # ------------------------------------------------------------------
    # Query – no filters
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Embed *query_text* and return the top-k nearest documents.

        Parameters
        ----------
        query_text : The natural-language query string.
        top_k      : Number of results to return.
        include    : ChromaDB fields to include in results.
                     Defaults to ["documents", "metadatas", "distances"].

        Returns
        -------
        List of result dicts, each with keys:
            id        – document id
            document  – raw document text
            metadata  – metadata dict
            distance  – distance score (lower = more similar for cosine/l2)
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        vector = self._embed(query_text)
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=include,
        )
        return self._format_results(raw)

    # ------------------------------------------------------------------
    # Query – simple equality filters (convenience wrapper)
    # ------------------------------------------------------------------

    def query_where(
        self,
        query_text: str,
        filters: dict[str, str | int | float | bool],
        top_k: int = 5,
        include: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convenience method for simple key=value metadata filters.

        Pass a plain dict and this method converts it to the correct
        ChromaDB ``where`` clause automatically.

        Parameters
        ----------
        query_text : The natural-language query string.
        filters    : Plain equality dict, e.g.::

                         {"source": "arxiv"}
                         {"source": "arxiv", "category": "ml"}   # AND

        top_k      : Number of results to return.

        Examples
        --------
        >>> store.query_where("transformers", {"source": "arxiv"})
        >>> store.query_where("attention", {"source": "arxiv", "year": 2023})
        """
        where = self._build_where(filters)
        return self.query_with_filters(
            query_text=query_text,
            where=where,
            top_k=top_k,
            include=include,
        )

    @staticmethod
    def _build_where(
        filters: dict[str, str | int | float | bool],
    ) -> dict[str, Any]:
        """
        Convert a flat equality dict into a ChromaDB ``where`` clause.

        - Single key  → ``{"key": {"$eq": value}}``
        - Multiple keys → ``{"$and": [{"k1": {"$eq": v1}}, ...]}``
        """
        if not filters:
            raise ValueError("filters dict must contain at least one key.")

        clauses = [{k: {"$eq": v}} for k, v in filters.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    # ------------------------------------------------------------------
    # Query – with raw ChromaDB filters
    # ------------------------------------------------------------------

    def query_with_filters(
        self,
        query_text: str,
        where: dict[str, Any],
        top_k: int = 5,
        include: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Embed *query_text* and return the top-k nearest documents that
        also satisfy *where* (a ChromaDB metadata filter).

        Parameters
        ----------
        query_text : The natural-language query string.
        where      : ChromaDB `where` filter dict.

                     Examples
                     --------
                     # Equality
                     {"source": "arxiv"}

                     # Range
                     {"year": {"$gte": 2022}}

                     # Logical AND
                     {"$and": [{"source": "arxiv"}, {"year": {"$gte": 2022}}]}

                     # Logical OR
                     {"$or": [{"category": "ml"}, {"category": "nlp"}]}

        top_k      : Number of results to return.
        include    : ChromaDB fields to include in results.

        Returns
        -------
        Same structure as :meth:`query`.
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        vector = self._embed(query_text)
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=include,
        )
        return self._format_results(raw)

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten ChromaDB's list-of-lists result structure."""
        ids        = (raw.get("ids")       or [[]])[0]
        documents  = (raw.get("documents") or [[]])[0]
        metadatas  = (raw.get("metadatas") or [[]])[0]
        distances  = (raw.get("distances") or [[]])[0]

        results = []
        for i, doc_id in enumerate(ids):
            results.append(
                {
                    "id":       doc_id,
                    "document": documents[i] if i < len(documents) else None,
                    "metadata": metadatas[i]  if i < len(metadatas) else {},
                    "distance": distances[i]  if i < len(distances) else None,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._collection.count()

    def health(self) -> dict[str, Any]:
        """Lightweight health-check payload."""
        return {
            "collection": self.collection_name,
            "document_count": self.count,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory,
        }
