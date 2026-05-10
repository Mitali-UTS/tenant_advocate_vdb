# Tenancy Documents Vector Search API

A semantic search API built with FastAPI and ChromaDB, enabling natural-language queries over a collection of tenancy regulation documents using sentence-transformer embeddings.

---

## About the Project

This API provides a simple interface for performing semantic (vector) search over tenancy-related documents stored in a ChromaDB collection. Documents are embedded using the `microsoft/harrier-oss-v1-0.6b` model and indexed for fast similarity search. The API supports plain queries as well as metadata-filtered queries using ChromaDB's native filter syntax or a simplified key=value interface.

**Tech stack:**
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [ChromaDB](https://www.trychroma.com/) — vector database
- [Sentence Transformers](https://www.sbert.net/) — embedding model (`microsoft/harrier-oss-v1-0.6b`)
- [Pydantic](https://docs.pydantic.dev/) — request/response validation

---

## Getting Started

### Prerequisites

- Python 3.11
- Git

### Clone & Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mitali-UTS/tenant_advocate_vdb.git
   cd tenant_advocate_vdb
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (optional)

   The app uses sensible defaults for local development, but you can override them:
   ```bash
   export COLLECTION_NAME="tenancy_documents"
   export PERSIST_DIRECTORY="vdb/tenancy_regulations"
   export EMBEDDING_MODEL="microsoft/harrier-oss-v1-0.6b"
   ```

5. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`.  
   Interactive docs (Swagger UI) are at `http://localhost:8000/docs`.

---

## API Reference

### `GET /health`
Returns the service status and collection statistics.

**Response**
```json
{
  "status": "ok",
  "collection": "tenancy_documents",
  "document_count": 1024
}
```

---

### `POST /query`
Semantic search with no filters. Embeds the query and returns the top-k most similar documents.

**Request body**
| Field   | Type    | Required | Default | Description                        |
|---------|---------|----------|---------|------------------------------------|
| `query` | string  | ✅       | —       | Natural-language search query      |
| `top_k` | integer | ❌       | 5       | Number of results to return (1–100)|

**Example request**
```json
{
  "query": "What are the rules for bond refunds?",
  "top_k": 3
}
```

**Response**
```json
{
  "query": "What are the rules for bond refunds?",
  "total": 3,
  "results": [
    {
      "id": "doc_001",
      "document": "Bond refunds must be processed within 14 days...",
      "metadata": { "source": "rta_qld", "year": 2022 },
      "distance": 0.142
    },
        {
      "id": "doc_002",
      "document": "Bond refunds must be processed within 14 days...",
      "metadata": { "source": "rta_qld", "year": 2022 },
      "distance": 0.242
    },
        {
      "id": "doc_003",
      "document": "Bond refunds must be processed within 14 days...",
      "metadata": { "source": "rta_qld", "year": 2022 },
      "distance": 0.342
    }
  ]
}
```

---

### `POST /query/filtered`
Semantic search with a full ChromaDB `where` metadata filter. Supports all ChromaDB operators.

**Request body**
| Field   | Type    | Required | Default | Description                         |
|---------|---------|----------|---------|-------------------------------------|
| `query` | string  | ✅       | —       | Natural-language search query       |
| `top_k` | integer | ❌       | 5       | Number of results to return (1–100) |
| `where` | object  | ✅       | —       | ChromaDB filter expression          |

**Supported operators:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`

**Example request**
```json
{
  "query": "notice period for ending a lease",
  "top_k": 5,
  "where": {
    "$and": [
      { "source": { "$eq": "rta_qld" } },
      { "year": { "$gte": 2020 } }
    ]
  }
}
```

**Response** — same shape as `/query`.

---

### `POST /query/where`
The simplest filtered search. Pass a plain key=value dict and the API handles ChromaDB filter syntax automatically. Multiple fields are combined with AND logic.

**Request body**
| Field     | Type    | Required | Default | Description                                    |
|-----------|---------|----------|---------|------------------------------------------------|
| `query`   | string  | ✅       | —       | Natural-language search query                  |
| `top_k`   | integer | ❌       | 5       | Number of results to return (1–100)            |
| `filters` | object  | ✅       | —       | Simple key=value metadata filters (AND logic)  |

**Example request**
```json
{
  "query": "repairs and maintenance obligations",
  "top_k": 5,
  "filters": {
    "source": "rta_qld",
    "year": 2023
  }
}
```

**Response** — same shape as `/query`.

---

### `POST /reload`
Hot-reloads the embedding model and/or reconnects to a new ChromaDB persist directory without restarting the server. Requires authentication if `API_KEY` is set.

**Request body**
| Field                  | Type   | Required | Description                                |
|------------------------|--------|----------|--------------------------------------------|
| `embedding_model_name` | string | ❌       | New model name to swap the embedding model |
| `persist_directory`    | string | ❌       | New path to reconnect the ChromaDB client  |

**Example request**
```json
{
  "embedding_model_name": "microsoft/harrier-oss-v1-0.6b",
  "persist_directory": "vdb/tenancy_regulations_v2"
}
```

**Response**
```json
{
  "status": "reloaded",
  "collection": "tenancy_documents",
  "document_count": 2048
}
```

---

## Search Result Schema

All search endpoints return results in the following format:

| Field      | Type    | Description                                        |
|------------|---------|----------------------------------------------------|
| `id`       | string  | Unique document ID in the ChromaDB collection      |
| `document` | string  | The document text chunk                            |
| `metadata` | object  | Key-value metadata associated with the document    |
| `distance` | float   | Embedding distance (lower = more similar)          |