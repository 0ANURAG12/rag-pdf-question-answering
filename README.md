# AI Service — RAG Pipeline (FastAPI)

This is the AI backend for the **AI-Driven Operations Improvement** project.
It accepts a PDF document, indexes its content, and answers questions about it
using a locally running language model — with no internet connection and no API keys required.

---

## Table of Contents

1. [Project Overview](#1-project-overview), [Key Features](#1a-key-features)
2. [What Problem This Solves](#2-what-problem-this-solves)
3. [What is RAG?](#3-what-is-rag)
4. [Why Each Technology Was Chosen](#4-why-each-technology-was-chosen)
5. [Technology Stack](#5-technology-stack)
6. [Folder Structure](#6-folder-structure)
7. [File Responsibilities](#7-file-responsibilities)
8. [System Architecture](#8-system-architecture)
9. [Complete RAG Pipeline](#9-complete-rag-pipeline)
10. [Single-Document Design](#10-single-document-design)
11. [Hallucination Prevention](#11-hallucination-prevention)
12. [API Endpoints](#12-api-endpoints)
13. [Installation and Setup](#13-installation-and-setup)
14. [Running the Project](#14-running-the-project)
15. [Example Workflow](#15-example-workflow)
16. [Configuration Reference](#16-configuration-reference)
17. [Current Limitations](#17-current-limitations)
18. [Future Improvements](#18-future-improvements)

---

## 1. Project Overview

This service is the AI brain of the system. It does one thing well:

> **Upload a PDF. Ask questions. Get accurate answers backed by the document.**

It is built as a standalone FastAPI service that can be called over HTTP from any client.
It can be tested independently using the built-in Swagger UI at `/docs`.

The entire AI pipeline runs locally on your machine:
- No OpenAI. No cloud APIs. No subscription fees.
- The language model runs through **Ollama**, a local model runner.
- Document vectors are stored in **ChromaDB**, a local vector database.

---

## 1a. Key Features

- **PDF Upload** — Upload any text-based PDF via a simple API call
- **PDF Text Extraction & Cleaning** — Extracts and normalizes text from every page
- **Intelligent Text Chunking** — Splits documents into overlapping chunks to preserve context; optional semantic chunking splits where meaning changes
- **Local Embedding Generation** — Uses Sentence Transformers (`all-MiniLM-L6-v2`) to convert text to vectors
- **Semantic Search** — ChromaDB finds the most relevant chunks by meaning, not keywords
- **Local LLM via Ollama** — Generates answers using `llama3.2:latest` running entirely on your machine
- **Retrieval-Augmented Generation (RAG)** — Grounds every answer in the actual document content
- **Hallucination Prevention** — The LLM is instructed to only answer from context; returns empty sources if nothing is found
- **Source References** — Every answer includes the exact chunks that were used to produce it
- **No API Keys Required** — Everything runs locally; no cloud services, no subscriptions
- **Structured Logging** — Single-line, request-correlated logs on every ingestion and query
- **Deep Health Check** — `GET /health/deep` verifies Ollama and ChromaDB are reachable, not just the web server
- **Retry + exponential backoff** — Ollama generation retries twice with doubling backoff before reporting an error
- **File Validation** — Rejects non-PDF uploads and corrupted PDFs with clear 400 errors
- **Tightened CORS** — Only origins listed in `ALLOWED_ORIGINS` may call the API
- **Lazy model loading** — The embedding model (`all-MiniLM-L6-v2`) is loaded on first request, not at server startup
- **Cross-Encoder Reranking** — Retrieved chunks are rescored by a cross-encoder for higher precision
- **Single-Document Design** — One active PDF at a time; uploading a new PDF replaces the previous one

---

## 2. What Problem This Solves

Imagine you have a 40-page operations report. You want to find specific information quickly.

Without AI:
- You read the whole document manually.
- You ctrl+F for keywords and hope the answer is nearby.

With this service:
- You upload the PDF once.
- You ask a question in plain English.
- The system finds the relevant parts of the document and generates a precise answer.

The key constraint: **the system only answers from the uploaded document**.
It will not make things up or use outside knowledge.

---

## 3. What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

It is a technique that combines two steps:

1. **Retrieval** — Find the most relevant sections of a document for a given question.
2. **Generation** — Give those sections to a language model and ask it to answer the question.

Without RAG, a language model answers from memory — which can be wrong, outdated, or fabricated.

With RAG, the model answers from the actual document — making it grounded and verifiable.

```
User Question
     │
     ▼
Search the document for relevant sections
     │
     ▼
Pass those sections + the question to the LLM
     │
     ▼
LLM reads only those sections and answers
     │
     ▼
Answer returned to user
```

---

## 4. Why Each Technology Was Chosen

### Why FastAPI?
FastAPI is a Python web framework. It was chosen because:
- The entire AI stack (PyPDF, Sentence Transformers, ChromaDB, Requests) is Python-native.
- It is fast, lightweight, and generates automatic API documentation at `/docs`.
- It works well as a microservice called by a Spring Boot backend.

### Why Ollama?
Ollama is a tool that runs large language models locally on your machine.
It was chosen because:
- No API key or internet connection is required.
- Models run on your own hardware — data stays private.
- It exposes a simple HTTP API (`/api/generate`) that is easy to call with `httpx`.
- Switching models is a one-line config change.

### Why ChromaDB?
ChromaDB is a vector database — a database that stores and searches embeddings (numeric representations of text).
It was chosen because:
- It runs entirely on disk, no separate server process needed.
- It supports metadata filtering (used to tag chunks by source filename).
- It is simple to set up with zero infrastructure.

### Why Sentence Transformers?
Sentence Transformers is a Python library for creating text embeddings.
An **embedding** is a list of numbers that represents the meaning of a piece of text.
Two pieces of text about the same topic will have similar numbers.

It was chosen because:
- The `all-MiniLM-L6-v2` model runs locally with no internet access.
- It is fast and produces high-quality embeddings for document search.
- It works well for matching questions to document chunks.

### Why PyPDF?
PyPDF is a Python library for reading PDF files.
It was chosen because it is lightweight, dependency-free, and handles standard PDF text extraction without external tools.

### Why LangChain Text Splitters?
LangChain's `RecursiveCharacterTextSplitter` was used only for its text splitting logic.
It splits text intelligently — trying paragraph boundaries first, then sentences, then words.
This produces semantically coherent chunks rather than cutting text at arbitrary character positions.

---

## 5. Technology Stack

| Technology | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| FastAPI | 0.115.0 | Web framework, API layer |
| Uvicorn | 0.30.6 | ASGI server (runs FastAPI) |
| PyPDF | 4.3.1 | PDF text extraction |
| LangChain Text Splitters | 0.3.0 | Text chunking |
| Sentence Transformers | 3.1.1 | Text embeddings (`all-MiniLM-L6-v2`) |
| ChromaDB | 0.5.5 | Vector storage and similarity search |
| Requests | 2.32.3 | HTTP calls to Ollama |
| Pydantic Settings | 2.5.2 | Environment variable configuration |
| python-multipart | 0.0.9 | File upload support |

---

## 6. Folder Structure

```
ai-service/
│
├── app/
│   ├── main.py                  # FastAPI app, middleware, request IDs, route registration
│   ├── config.py                # All settings loaded from .env
│   ├── exceptions.py            # Custom exceptions and error handlers
│   ├── schemas.py               # Request/response data models
│   ├── logging_config.py        # Structured single-line / JSON logging setup
│   │
│   ├── routes/
│   │   ├── ingest.py            # POST /api/ingest — PDF upload endpoint
│   │   └── query.py             # POST /api/query — question answering endpoint
│   │
│   └── services/
│       ├── pdf_service.py       # Stage 1: Extract and clean text from PDF
│       ├── chunking_service.py  # Stage 2: Split text into chunks
│       ├── embedding_service.py # Stage 3: Convert text to embeddings
│       ├── vectorstore_service.py # Stage 4: Store and search embeddings in ChromaDB
│       ├── retrieval_service.py # Stage 5: Find relevant chunks for a question
│       ├── rerank_service.py    # Stage 5b: Rerank candidates by question–chunk relevance
│       ├── prompt_service.py    # Stage 6: Build the prompt for the LLM
│       └── llm_service.py       # Stage 7: Send prompt to Ollama, return answer
│
├── chroma_data/                 # ChromaDB persisted storage (auto-created)
├── .env                         # Your local configuration (not committed)
├── .env.example                 # Template showing all available config keys
├── run.sh                       # One-command launcher (activates venv, starts server)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 7. File Responsibilities

### `app/main.py`
Creates the FastAPI application. Sets up structured logging, registers CORS middleware
(restricted to the origins in `ALLOWED_ORIGINS`), adds a request-ID middleware that tags every
request with an `X-Request-Id` header and logs it, registers custom exception handlers, and
includes the two API routers. Exposes `GET /health` for uptime checks and `GET /health/deep`
which also verifies Ollama and ChromaDB are reachable. Closes the shared httpx client on shutdown.

### `app/config.py`
Reads all configuration from environment variables or `.env`.
Creates a single `settings` object used across the entire app.
Covers: ChromaDB path, collection name, embedding model, chunk size, chunk overlap, top_k, Ollama URL, model name, and timeout.

### `app/exceptions.py`
Defines four custom exceptions:
- `EmptyDocumentError` — PDF has no extractable text.
- `InvalidPdfError` — Uploaded file is not a readable PDF (wrong type, bad magic bytes, or corrupted).
- `NoDocumentsIndexedError` — Query attempted before any PDF was uploaded.
- `OllamaServiceError` — Ollama is unreachable or returned an empty response.

Each exception maps to a clean JSON error response with an appropriate HTTP status code.
A generic handler catches any unexpected errors with status 500.

### `app/schemas.py`
Defines the data shapes for API requests and responses:
- `IngestResponse` — Returned after PDF upload (filename, chunks added, message).
- `QueryRequest` — Incoming question and optional `top_k`.
- `QueryResponse` — Answer text and list of source chunks.
- `SourceChunk` — One retrieved chunk: content, source filename, chunk index.

### `app/routes/ingest.py`
Handles `POST /api/ingest`. Reads the uploaded file bytes, runs the full ingestion pipeline
(extract → chunk → embed → store), and returns an `IngestResponse`.

### `app/routes/query.py`
Handles `POST /api/query`. Retrieves relevant chunks, builds the prompt, calls Ollama,
and returns the answer. If Ollama responds with the "not found" message, sources are returned as empty.

### `app/services/pdf_service.py`
Extracts raw text from a PDF using PyPDF.
After extraction, cleans the text: normalizes line endings, merges soft-wrapped lines into spaces,
preserves paragraph breaks, and collapses repeated whitespace.
Raises `EmptyDocumentError` if no text is found (e.g. scanned image PDFs).

### `app/services/chunking_service.py`
Splits the cleaned document text into overlapping chunks. Two strategies are supported, selected by `CHUNK_STRATEGY`:

- **`recursive`** (default): LangChain's `RecursiveCharacterTextSplitter`, which breaks on paragraph boundaries first, then sentences, then words, with a fixed character overlap. Default: 500 characters per chunk, 50 character overlap. Overlap ensures context is not lost at chunk boundaries.
- **`semantic`**: splits where the meaning of the document actually changes. Each sentence is embedded with the same model used for retrieval, and a new chunk starts when adjacent sentences become too dissimilar (cosine distance above `SEMANTIC_THRESHOLD`), still capped at `chunk_size` characters. This keeps semantically coherent content together instead of cutting it at arbitrary character positions.

### `app/services/embedding_service.py`
Loads the `all-MiniLM-L6-v2` model once at startup.
Provides two functions:
- `embed_texts()` — embeds a list of chunks (used during ingestion).
- `embed_query()` — embeds a single question (used during retrieval).

### `app/services/vectorstore_service.py`
Manages ChromaDB. On ingestion: clears the entire collection first (single-document design), then adds new chunks with UUID identifiers and metadata (source filename, chunk index). On query: searches for the `top_k` most similar chunks using cosine similarity.

### `app/services/retrieval_service.py`
Coordinates retrieval: checks that documents exist, resolves `top_k` (uses config default if not provided or if `<= 0`), embeds the question, fetches extra candidates from the vector store, and passes them through the reranker.

### `app/services/rerank_service.py`
Rescores the candidate chunks with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that scores each (question, chunk) pair jointly, then keeps only the `top_k` most relevant. This catches chunks the bi-encoder missed because they only match the question in combination with other words. Falls back to the original retrieval order if the reranker fails, so retrieval never breaks because of reranking. Disabled entirely when `RERANK_ENABLED=false`.

### `app/services/prompt_service.py`
Builds the prompt sent to Ollama. Each retrieved chunk is labelled with its source file and chunk index. The LLM is explicitly instructed to answer only from the provided context and to say "I could not find this information in the uploaded documents." if the answer is not present.

### `app/services/llm_service.py`
Sends the prompt to Ollama's `POST /api/generate` endpoint with `stream: false` using a shared
`httpx.AsyncClient` (non-blocking, so the FastAPI event loop is never held up). Also exposes
`list_models()` for health checks. Raises `OllamaServiceError` if the request fails or the
response is empty.

---

## 8. System Architecture

```
        User (Swagger UI / Client)
                   │
                   │  HTTP Request
                   ▼
       ┌─────────────────────┐
       │   FastAPI Service   │
       │  (ai-service:8000)  │
       └──────────┬──────────┘
                  │
    ┌─────────────┴──────────────┐
    │                            │
    ▼                            ▼
POST /api/ingest          POST /api/query
(PDF Upload)              (Ask a Question)
    │                            │
    ▼                            │
┌────────────────┐               │
│ PDF Extraction │               │
│  & Cleaning    │               │
└───────┬────────┘               │
        │                        ▼
        ▼               ┌──────────────────┐
┌────────────────┐       │  Embed Question  │
│   Chunking     │       └────────┬─────────┘
└───────┬────────┘                │
        │                        ▼
        ▼               ┌──────────────────┐
┌────────────────┐       │ ChromaDB Search  │
│  Embeddings    │       │  (top-k chunks)  │
│ (all-MiniLM)  │       └────────┬─────────┘
└───────┬────────┘                │
        │                        │
        ▼                        │
┌──────────────────────────────────────────┐
│           ChromaDB (on disk)             │
│   Stores chunk vectors + metadata        │
└──────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Prompt Builder  │
                        │ (context +      │
                        │  question)      │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Ollama (local) │
                        │ llama3.2:latest │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │    Response     │
                        │ answer + sources│
                        └─────────────────┘
```

---

## 9. Complete RAG Pipeline

### Stage 1 — PDF Extraction (`pdf_service.py`)

**What it does:** Reads the PDF file bytes and extracts all text from every page using PyPDF.

**Why it is necessary:** All downstream processing works on plain text. The PDF format stores text as layout instructions, not readable strings. PyPDF converts those instructions into text.

**Text cleaning:** PyPDF outputs text exactly as it appears in the PDF layout — including a newline at every line wrap. Without cleaning, words run together (e.g. `DataScience\nStudent` becomes `DataScienceStudent`). The cleaning step:
1. Normalizes line endings (`\r\n` → `\n`)
2. Protects paragraph breaks (2+ newlines)
3. Merges single newlines into spaces (fixing split words)
4. Collapses repeated spaces
5. Trims each paragraph

---

### Stage 2 — Chunking (`chunking_service.py`)

**What it does:** Splits the cleaned document text into small, overlapping pieces called **chunks**.

**Why it is necessary:** Language models have a limited input size. A 40-page document cannot be sent to the model all at once. Chunking breaks it into pieces small enough to fit. Chunks are also the unit of search — retrieval finds the most relevant chunks, not the most relevant pages.

**How it works:**
- Uses `RecursiveCharacterTextSplitter` which tries to split on paragraph breaks first, then sentences, then words.
- Default chunk size: 500 characters.
- Overlap: 50 characters. This means the last 50 characters of one chunk also appear at the start of the next, preventing loss of context at boundaries.
- Optional `semantic` strategy (`CHUNK_STRATEGY=semantic`) splits where meaning changes instead: each sentence is embedded, and a new chunk starts when adjacent sentences are too dissimilar (cosine distance above `SEMANTIC_THRESHOLD`), still capped at `chunk_size` characters.

**Why overlap matters — example:**

Without overlap, two adjacent chunks might look like:
```
Chunk 1: "The company reduced operational"
Chunk 2: "costs using AI automation."
```
The sentence is split. If someone asks about cost reduction, neither chunk alone contains the full sentence. Retrieval may miss the relevant context.

With 50-character overlap:
```
Chunk 1: "The company reduced operational costs"
Chunk 2: "operational costs using AI automation."
```
Both chunks now carry enough context for the retrieval to find them.

---

### Stage 3 — Embeddings (`embedding_service.py`)

**What it does:** Converts each chunk of text into a list of numbers called an **embedding** (or vector).

**Why it is necessary:** Computers cannot compare text meaning directly. But they can compare numbers. An embedding turns "the company reduced costs by 20%" into a list like `[0.12, -0.45, 0.88, ...]`. Two texts with similar meaning will produce similar numbers. This is what makes semantic search possible.

**Model used:** `all-MiniLM-L6-v2` from Sentence Transformers.
- Produces **384-dimensional vectors** — each text is represented as a list of 384 numbers.
- Runs entirely locally with no internet access required.
- Selected for its balance of speed and quality: it is significantly faster than larger models while still producing accurate semantic embeddings for document retrieval tasks.
- The same model is used for both document chunks (at ingestion) and questions (at query time), which ensures the embeddings are comparable.

---

### Stage 4 — Vector Storage (`vectorstore_service.py`)

**What it does:** Stores all chunk embeddings in ChromaDB along with metadata: the source filename and the chunk's position index.

**Why it is necessary:** Embeddings must be stored so they can be searched later when a question arrives. ChromaDB stores them on disk and provides fast similarity search.

**Single-document design:** When a new PDF is uploaded, all existing vectors are deleted first. This ensures queries always search only the currently uploaded document.

---

### Stage 5 — Retrieval (`retrieval_service.py` + `vectorstore_service.py`)

**What it does:** When a question arrives, it is embedded using the same model. ChromaDB then finds the `top_k` stored chunks whose embeddings are most similar to the question embedding.

**Why it is necessary:** Sending all chunks to the LLM would be slow and exceed the model's context limit. Retrieval selects only the most relevant pieces.

**top_k:** Controls how many chunks are retrieved. Default is 4 (from config). Can be overridden per request. If the provided value is `<= 0` or not provided, the config default is used.

**Reranking:** Retrieval first fetches more candidates than needed (default 12), then `rerank_service.py` rescores each (question, chunk) pair with a cross-encoder and keeps only the `top_k` most relevant. The bi-encoder used for retrieval encodes the question and each chunk independently, so it can miss chunks that only match the question in combination with other words. The cross-encoder scores them jointly, which is far more accurate — at the cost of speed, which is why it only runs on the small candidate set.

**What is Semantic Search?**
Traditional keyword search looks for exact word matches. If your document says "cost reduction" and you search for "saving money", a keyword search finds nothing.

Semantic search finds matches by meaning. The question "How was money saved?" and the text "operational costs were reduced by 20%" produce similar embeddings — so ChromaDB will find that chunk even without matching words. This makes retrieval far more useful for natural language questions.

ChromaDB compares the question embedding against all stored chunk embeddings using **cosine similarity** — a mathematical measure of how close two vectors point in the same direction. The chunks with the highest similarity scores are returned as the most relevant results.

---

### Stage 6 — Prompt Construction (`prompt_service.py`)

**What it does:** Builds the text prompt that will be sent to the language model. Each retrieved chunk is labelled with its source file and chunk index. The full prompt looks like:

```
Answer only from the provided context.

If the answer is not in the context, reply exactly:
"I could not find this information in the uploaded documents."

Do not guess.

Context:
[Source: report.pdf | Chunk: 2]
... chunk content here ...

[Source: report.pdf | Chunk: 5]
... chunk content here ...

Question:
What does the document say about cost reduction?

Answer:
```

**Why it is necessary:** Without explicit instructions, language models use their own knowledge to fill in gaps. The prompt is the contract: the model is told exactly where to look and what to say when the answer is not there.

---

### Stage 7 — LLM Generation (`llm_service.py`)

**What it does:** Sends the constructed prompt to Ollama's local HTTP API and returns the generated answer.

**Why it is necessary:** This is where the actual answer is produced. The model reads the context chunks and synthesizes a coherent, direct answer.

**How it works:** Calls `POST http://localhost:11434/api/generate` with `stream: false`. The full response is returned in one JSON payload. The `response` field contains the answer text.

**Model used:** `llama3.2:latest` — configured in `.env` via `OLLAMA_MODEL`. This can be changed to any model available in Ollama without modifying code.

---

### Stage 8 — Response Assembly (`routes/query.py`)

**What it does:** Returns the answer and the source chunks to the caller.

**Special case:** If Ollama's answer is exactly `"I could not find this information in the uploaded documents."`, the sources list is returned empty. This prevents the API from returning misleading source references when the LLM found nothing useful.

---

### Full Data Flow Diagram

```
PDF Upload (POST /api/ingest)
──────────────────────────────────────────────────────────────
PDF bytes
  │
  ▼ pdf_service.extract_text()
Raw text (cleaned)
  │
  ▼ chunking_service.split_text()
["chunk 1 text", "chunk 2 text", ...]
  │
  ▼ embedding_service.embed_texts()
[[0.12, -0.45, ...], [0.33, 0.71, ...], ...]
  │
  ▼ vectorstore_service.add_chunks()
ChromaDB (clears old, stores new) ──── Persisted to disk
  │
  ▼
IngestResponse { filename, chunks_added, message }


Question (POST /api/query)
──────────────────────────────────────────────────────────────
"What does the document say about X?"
  │
  ▼ embedding_service.embed_query()
[0.22, -0.31, ...]   ← question embedding
  │
  ▼ vectorstore_service.query()
Top 4 most similar chunks from ChromaDB
  │
  ▼ prompt_service.build_prompt()
Full prompt with context + question
  │
  ▼ llm_service.generate_answer()
"The document states that X..."
  │
  ▼ routes/query.py
QueryResponse { answer, sources: [...] }
```

---

## 10. Single-Document Design

This service is designed for **one document at a time**.

When a new PDF is uploaded:
1. ChromaDB is completely cleared.
2. The new PDF is indexed from scratch.
3. All queries now search only the new document.

**Why this design?**
This is a prototype. Supporting multiple documents simultaneously requires document selection logic, user sessions, and more complex retrieval filtering. Those features add complexity without adding value to a single-user demonstration. The design is intentionally simple and predictable.

**ChromaDB is kept** (not replaced with in-memory storage) because:
- It persists data across server restarts.
- It provides production-quality vector search.
- It can support multi-document retrieval in the future with minimal changes.

---

## 11. Hallucination Prevention

A **hallucination** is when a language model invents information that is not in the source material. This is a well-known problem with LLMs.

This service prevents hallucinations through two mechanisms:

**1. Prompt-level constraint:**
The prompt explicitly tells the model:
```
Answer only from the provided context.
If the answer is not in the context, reply exactly:
"I could not find this information in the uploaded documents."
Do not guess.
```

**2. Response-level check:**
`query.py` checks if the answer equals the exact "not found" message.
If it does, `sources: []` is returned — no source chunks are fabricated.

Although RAG greatly reduces hallucinations, no language model can guarantee perfect accuracy. The quality of the final answer depends on the quality of the retrieved document chunks.

---

## 12. API Endpoints

### `POST /api/ingest`

Upload a PDF document.

**Request:** `multipart/form-data`
```
file: <PDF file>
```

**Response:**
```json
{
  "filename": "report.pdf",
  "chunks_added": 8,
  "message": "Document processed and indexed successfully."
}
```

**Errors:**
- `400` — Uploaded file is not a PDF (wrong MIME type or missing `%PDF-` magic bytes).
- `400` — PDF is corrupted or unreadable.
- `400` — PDF contains no extractable text (scanned image without text layer).

---

### `POST /api/query`

Ask a question about the uploaded document.

**Request:**
```json
{
  "question": "What does the document say about cost reduction?",
  "top_k": 4
}
```

`top_k` is optional. If omitted or `<= 0`, the config default (4) is used.
`question` is limited to 2000 characters; `top_k` must be between 1 and 50.

**Response (answer found):**
```json
{
  "answer": "The document states that operational costs were reduced by 20% through process automation.",
  "sources": [
    {
      "content": "Phase 2 focused on automation which reduced costs by 20%...",
      "source": "report.pdf",
      "chunk_index": 3,
      "distance": 1.31
    }
  ]
}
```

Every response also carries an `X-Request-Id` header for tracing the request through the logs.

**Response (answer not found):**
```json
{
  "answer": "I could not find this information in the uploaded documents.",
  "sources": []
}
```

**Errors:**
- `400` — No document has been uploaded yet.
- `502` — Ollama is not running or the model is not pulled.

---

### `GET /health`

Check if the service is running.

**Response:**
```json
{ "status": "ok" }
```

### `GET /health/deep`

Check that every dependency the service relies on is reachable — the local
Ollama server and the ChromaDB store. Returns `200` when all dependencies
are healthy, `503` when any are not, with a per-dependency status in the
body.

**Response:**
```json
{
  "status": "ok",
  "dependencies": {
    "ollama": { "models": ["llama3.2:latest", "nomic-embed-text:latest"] },
    "chromadb": { "document_count": 10 }
  }
}
```

---

## 13. Installation and Setup

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed on your machine
- A model pulled in Ollama

```bash
# Install Ollama, then pull a model
ollama pull llama3.2
```

### Setup

```bash
# Navigate to the ai-service directory
cd ai-service

# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Create your .env file from the template
cp .env.example .env
```

Open `.env` and adjust values if needed. The defaults work out of the box for local development.

---

## 14. Running the Project

Make sure Ollama is running first:
```bash
ollama serve
```

Then start the FastAPI service:
```bash
uvicorn app.main:app --reload --port 8000
```

Or use the bundled launcher (activates the venv, installs deps, starts on port 8000):
```bash
./run.sh
```

The service starts at: `http://localhost:8000`

Swagger UI (interactive API docs): `http://localhost:8000/docs`

---

## 15. Example Workflow

### Using Swagger UI

1. Open `http://localhost:8000/docs`
2. Click `POST /api/ingest` → `Try it out`
3. Upload a PDF file
4. Note the `chunks_added` count in the response
5. Click `POST /api/query` → `Try it out`
6. Enter your question:
   ```json
   { "question": "What are the expected outcomes of this project?" }
   ```
7. Read the answer and the source chunks it was based on

### Using curl

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@/path/to/your/document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main objectives?"}'
```

---

## 16. Configuration Reference

All values can be set in `.env` or as environment variables.

| Key | Default | Description |
|---|---|---|
| `CHROMA_DIR` | `./chroma_data` | Where ChromaDB stores its files |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model name |
| `CHUNK_STRATEGY` | `recursive` | Chunking strategy: `recursive` or `semantic` |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Characters shared between adjacent chunks |
| `SEMANTIC_THRESHOLD` | `0.5` | Cosine distance above which semantic chunking starts a new chunk |
| `TOP_K` | `4` | Default number of chunks to retrieve |
| `RERANK_ENABLED` | `true` | Whether to rerank retrieved candidates with a cross-encoder |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model used for reranking |
| `RERANK_CANDIDATES` | `12` | Number of candidates retrieved before reranking |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:latest` | Model name to use for generation |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Request timeout for Ollama calls |
| `ALLOWED_ORIGINS` | `http://localhost:8080` | Comma-separated list of CORS origins allowed to call the API |
| `LOG_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `JSON_LOGS` | `false` | Set to `true` to emit single-line JSON logs for log aggregators |
| `OLLAMA_MAX_RETRIES` | `2` | Retries for Ollama generation before failing |
| `OLLAMA_BASE_BACKOFF_SECONDS` | `2.0` | Base backoff seconds (doubles each retry) |

---

## 17. Current Limitations

These are known limitations of the current implementation:

- **Single PDF only** — Only one document can be active at a time. Uploading a new PDF removes the previous one from the index.
- **No OCR support** — Scanned PDFs (images of text without a selectable text layer) cannot be processed. The service will return an error for these files.
- **Local execution only** — The LLM runs through Ollama on your local machine. The service does not support cloud-hosted models out of the box.
- **Retrieval quality depends on PDF text quality** — If the PDF has poor formatting, unusual encoding, or complex multi-column layouts, the extracted text may be incomplete or incorrectly ordered, which reduces answer quality.
- **File validation is minimal** — The service checks the MIME type and PDF magic bytes, but does not verify that the PDF is structurally valid beyond what PyPDF can read. Corrupted PDFs are rejected with a clear 400 error.

---

## 18. Future Improvements

These are not implemented in the current version but are straightforward to add:

**Multi-document support**
Remove the collection-clear behavior. Add a `source` filter to ChromaDB queries so each query targets a specific document. Requires tracking which document is active per session or user.

**File type validation**
Currently any file can be uploaded to `/api/ingest`. Adding a MIME type check (`application/pdf`) would prevent unhelpful errors from non-PDF files.

**Streaming responses**
Ollama supports streaming generation (`stream: true`). Enabling this would allow the answer to appear word by word in the frontend instead of waiting for the full response.

**Logging**
Add Python's `logging` module to record ingestion counts, query latency, and errors. Currently the service has no application-level logs.

**Ollama health check**
The `/health` endpoint only confirms FastAPI is running. Adding an Ollama ping check would give a more accurate health signal for the full system.

**Rate limiting**
No rate limiting is currently applied. For a production deployment, adding per-IP or per-session limits would prevent abuse.

**Chunk quality improvements**
PDF extraction quality varies by PDF. Some documents (especially scanned PDFs or those with complex layouts) may produce poor chunks. Exploring alternative extraction libraries or OCR integration could improve accuracy for difficult documents.
