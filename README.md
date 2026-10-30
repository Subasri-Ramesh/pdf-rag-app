# PDF Question Answering with RAG

A complete Streamlit application that answers questions from uploaded PDFs using
page-aware retrieval-augmented generation (RAG). It extracts embedded text with
PyMuPDF, creates local Sentence Transformers embeddings, stores vectors in a
session-scoped ChromaDB collection, retrieves relevant passages, and asks
DeepSeek to answer strictly from those passages.

## Features

- **Multi-File Search**: Direct vector & semantic search across multiple files simultaneously with similarity scores, keyword highlighting, and per-document result breakdowns.
- **Cross-Document Question Answering**: Synthesizes information across multiple uploaded documents with balanced retrieval, conflict detection, and exact page citations `[filename.pdf, p. N]`.
- **Flexible Scope Filtering**: Seamlessly toggle between querying all uploaded files or focusing on specific documents for targeted comparison.
- **Batch Multi-PDF Ingestion**: Upload and index multiple PDFs in one operation with extension, signature, size, corruption, encryption, and empty-file validation.
- **Detection of Image-Only/Scanned PDFs**: Automatically identifies PDFs needing OCR and provides actionable feedback without crashing.
- **One-Based Page Metadata**: Chunks never cross page boundaries, guaranteeing unambiguous page-level citations.
- **Local `all-MiniLM-L6-v2` Embeddings**: Computes normalized vectors locally with cosine distance in ChromaDB.
- **Deterministic Insufficient-Evidence Fallback**: Immediate fallback response without unnecessary API calls when relevant evidence is lacking.
- **In-Memory Session Vector Store**: Complete session isolation with ephemeral in-memory Chroma storage.
- **Document Explorer & Transparency**: Inspect extracted chunks and verify page boundaries directly from the UI.
- **DeepSeek API Generation**: High-quality answers strictly grounded in retrieved evidence.
- **Comprehensive Unit Tests**: Fully mocked test suite using generated PDFs and deterministic test doubles.

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
                          fixed fallback              DeepSeek
                                                         │
                                                         ▼
                                            grounded answer + citations
```

`app.py` owns the Streamlit UI and per-session state. `ingest.py` is responsible
for all untrusted-file handling and page-aware chunks. `rag.py` owns vector
storage, retrieval, evidence formatting, and DeepSeek generation. Chunks never
cross page boundaries, so every citation maps to exactly one PDF page.

The Chroma client is ephemeral by design. Closing/expiring a Streamlit session
removes its vectors; no uploaded PDF bytes are written to disk. The cached
Sentence Transformer model is shared safely because it contains model weights,
not user documents.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Internet access on first launch to download the Sentence Transformers model
- A DeepSeek API key

## Setup and run

### Windows PowerShell

```powershell
cd C:\Users\TSI\Documents\Codex\2026-10-20\exp\outputs\pdf-rag-app
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Set your DEEPSEEK_API_KEY in .env
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
# Set your DEEPSEEK_API_KEY in .env
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
Upload one or more text-based PDFs, select **Process selected PDFs**, and use the
chat box. For scanned PDFs, run OCR in a PDF tool first and upload the resulting
searchable PDF.

## Run tests

With the virtual environment activated:

```powershell
python -m pytest -q
```

The tests do not require an active API key or network access, and do not
download a Sentence Transformers model. They use generated PDFs and test doubles.

## Configuration

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `DEEPSEEK_API_KEY` | Yes | none | DeepSeek API key |
| `DEEPSEEK_MODEL` | No | `deepseek-v4-flash` | DeepSeek model name (`deepseek-v4-flash`, etc.) |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com` | DeepSeek API base URL |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |

Do not put secrets in source code or commit `.env`. In production, inject the
variables with the hosting platform's secret manager (such as Streamlit Community Cloud Secrets).

## Multi-File Search and Grounding Behavior

### Multi-File Search
The **Search Across Files** tab provides direct vector search across all indexed PDFs (or a filtered selection). Users can adjust the minimum similarity threshold and maximum results. Matches are displayed with:
- Filename and exact page number
- Similarity percentage match
- Excerpt preview with query term highlighting
- Quick action button to ask a grounded question about that specific excerpt in the Q&A tab

### Multi-File Retrieval and Answering
When answering questions across multiple documents:
1. **Scope Selection**: Users can query across all indexed documents or select specific files from the scope filter.
2. **Balanced Retrieval**: To avoid a single document monopolizing all retrieved passages, the retriever queries a candidate pool and balances representation across all relevant documents before ordering chunks by similarity.
3. **Thresholding**: Cosine distance cutoff of `0.75` filters out irrelevant noise. If no chunk meets this threshold, the deterministic message is returned immediately without calling the API:
   `I don't have enough evidence in the uploaded PDFs to answer that.`
4. **Cross-Document Synthesis**: If sufficient evidence exists, DeepSeek is prompted to synthesize findings across the files, explicitly compare or contrast source documents when appropriate, report conflicts, and provide exact citations `[filename.pdf, p. N]` for every factual claim.

