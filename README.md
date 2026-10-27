# PDF Question Answering with RAG

A complete Streamlit application that answers questions from uploaded PDFs using
page-aware retrieval-augmented generation (RAG). It extracts embedded text with
PyMuPDF, creates local Sentence Transformers embeddings, stores vectors in a
session-scoped ChromaDB collection, retrieves relevant passages, and asks a
local Ollama model to answer strictly from those passages. OpenAI remains an
optional provider, but the default setup requires no API key or paid credits.

## Features

- Multiple PDF uploads with extension, signature, size, corruption, encryption,
  and empty-file validation
- Detection of image-only/scanned PDFs that need OCR
- One-based PDF page metadata retained through overlapping chunking
- Local `all-MiniLM-L6-v2` embeddings and cosine similarity search in ChromaDB
- Evidence-only responses with `[filename.pdf, p. N]` citations
- A deterministic insufficient-evidence fallback before calling the model
- In-memory vector stores isolated per Streamlit browser session
- Per-document deletion, duplicate detection, and visible retrieved evidence
- Free local answer generation through Ollama
- Optional OpenAI provider with keys read only from environment variables
- Unit tests with generated PDFs and fake Ollama/OpenAI clients

## Architecture

```text
PDF upload
   │
   ▼
ingest.py ── validate ── extract per page ── overlapping page-local chunks
   │
   ▼
Sentence Transformer ── normalized embeddings
   │
   ▼
Session-only ChromaDB collection (cosine distance)
   │
   ├── delete by document_id
   │
Question ── embed ── retrieve top matching chunks ── similarity cutoff
                                                     │
                                  insufficient ──────┴─── sufficient
                                      │                       │
                                      ▼                       ▼
                              fixed fallback      Ollama (default) / OpenAI
                                                             │
                                                             ▼
                                                grounded answer + citations
```

`app.py` owns the Streamlit UI and per-session state. `ingest.py` is responsible
for all untrusted-file handling and page-aware chunks. `rag.py` owns vector
storage, retrieval, evidence formatting, and local/API generation. Chunks never
cross page boundaries, so every citation maps to exactly one PDF page.

The Chroma client is ephemeral by design. Closing/expiring a Streamlit session
removes its vectors; no uploaded PDF bytes are written to disk. The cached
Sentence Transformer model is shared safely because it contains model weights,
not user documents.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- [Ollama for Windows](https://ollama.com/download/windows), macOS, or Linux
- Internet access on first launch to download the Sentence Transformers model
  and the selected Ollama model

## Setup and run

### Windows PowerShell

```powershell
cd C:\Users\TSI\Documents\Codex\2026-10-20\exp\outputs\pdf-rag-app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
python -m streamlit run app.py
```

### macOS or Linux

```bash
cd /path/to/pdf-rag-app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
Upload one or more text-based PDFs, select **Process selected PDFs**, and use the
chat box. For scanned PDFs, run OCR in a PDF tool first and upload the resulting
searchable PDF.

Ollama runs in the background on Windows and serves its local API at
`http://localhost:11434`. If the application cannot connect, open Ollama from
the Start menu or run `ollama serve` in a separate terminal. The application
uses Ollama's documented [`POST /api/chat`](https://docs.ollama.com/api/chat)
endpoint.

## Run tests

With the virtual environment activated:

```powershell
python -m pytest -q
```

The tests do not require an API key, do not call Ollama or OpenAI, and do not
download a Sentence Transformers model. They use generated PDFs and test doubles.

## Configuration

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `LLM_PROVIDER` | No | `deepseek` | Generation provider: `deepseek`, `ollama`, or `openai` |
| `DEEPSEEK_API_KEY` | Only for DeepSeek | none | DeepSeek API key |
| `DEEPSEEK_MODEL` | No | `deepseek-v4-flash` | DeepSeek model name (`deepseek-v4-flash`, `deepseek-flash`, etc.) |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com` | DeepSeek API base URL |
| `OLLAMA_MODEL` | No | `llama3.2:3b` | Local Ollama generation model |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Local Ollama API URL |
| `OPENAI_API_KEY` | Only for OpenAI | none | Optional OpenAI credential |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Optional OpenAI generation model |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |

Do not put secrets in source code or commit `.env`. In production, inject the
variables with the hosting platform's secret manager and serve Streamlit behind
TLS. Because session state is process-local, a multi-worker deployment should
add a server-side session store and authenticated tenancy before using persistent
Chroma storage.

## Retrieval and grounding behavior

The retriever returns up to five chunks whose cosine distance is at most `0.75`.
If none pass, the app returns: `I don't have enough evidence in the uploaded PDFs
to answer that.` without calling a language model. If evidence passes, the
selected model receives only the question and labeled evidence, with instructions to avoid outside
knowledge, cite claims, report conflicts, and use the same fallback if the
passages still do not support an answer.

Similarity thresholds depend on the document domain. For production, evaluate
retrieval on representative questions before adjusting `max_distance` in
`RAGStore.retrieve`.
