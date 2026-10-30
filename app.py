"""Streamlit user interface for multi-file search and page-cited question answering."""

from __future__ import annotations

import html
import os
import re
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ingest import PDFIngestionError, ScannedPDFError, ingest_pdf
from rag import (
    INSUFFICIENT_EVIDENCE,
    RAGStore,
    answer_question,
)

try:
    from rag import summarize_retrieval_by_document
except ImportError:
    def summarize_retrieval_by_document(chunks):
        counts = {}
        for chunk in chunks:
            counts[getattr(chunk, "filename", "")] = counts.get(getattr(chunk, "filename", ""), 0) + 1
        return counts



load_dotenv()
st.set_page_config(
    page_title="Multi-File PDF Search & Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1f77b4, #00c0f2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .scope-banner {
        background-color: rgba(31, 119, 180, 0.08);
        border-left: 4px solid #1f77b4;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.35rem;
    }
    .badge-file {
        background-color: #e2e8f0;
        color: #1e293b;
    }
    .badge-page {
        background-color: #dbeafe;
        color: #1e40af;
    }
    .badge-score {
        background-color: #dcfce7;
        color: #166534;
    }
    .search-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.85rem;
        background: rgba(255, 255, 255, 0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def initialize_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_store" not in st.session_state:
        embedding_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            st.session_state.rag_store = RAGStore(
                load_embedding_model(embedding_name), st.session_state.session_id
            )
        except Exception:
            try:
                from chromadb.api.shared_system_client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            st.session_state.rag_store = RAGStore(
                load_embedding_model(embedding_name), st.session_state.session_id
            )
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def delete_document(document_id: str) -> None:
    st.session_state.rag_store.delete_document(document_id)
    st.session_state.documents.pop(document_id, None)


def clear_all_documents() -> None:
    st.session_state.rag_store.clear()
    st.session_state.documents.clear()
    st.session_state.messages.clear()


def highlight_query(text: str, query: str) -> str:
    """Highlight matched query words in snippet safely."""
    escaped = html.escape(text)
    words = [re.escape(w) for w in query.strip().split() if len(w) > 2]
    if not words:
        return escaped
    pattern = re.compile(rf"({'|'.join(words)})", re.IGNORECASE)
    return pattern.sub(
        r"<mark style='background-color: #fef08a; color: #111; padding: 1px 4px; border-radius: 3px;'>\1</mark>",
        escaped,
    )


initialize_session()

# Sidebar: Document uploads and multi-file management
with st.sidebar:
    st.header("📚 Document Library")
    st.caption("Upload multiple text-based PDFs to search and answer questions across all of them.")

    uploads = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDFs (max 25 MB each).",
    )

    if st.button("⚡ Process Selected PDFs", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Please choose at least one PDF file first.")
        else:
            success_count = 0
            for upload in uploads:
                try:
                    document = ingest_pdf(upload.name, upload.getvalue())
                    if document.document_id in st.session_state.documents:
                        st.info(f"ℹ️ {document.filename} is already indexed.")
                        continue
                    with st.spinner(f"Indexing {document.filename}…"):
                        chunk_count = st.session_state.rag_store.add_document(document)
                    st.session_state.documents[document.document_id] = {
                        "filename": document.filename,
                        "pages": document.page_count,
                        "chunks": chunk_count,
                        "raw_chunks": [
                            {"page": c.page_number, "text": c.text}
                            for c in document.chunks
                        ],
                    }
                    success_count += 1
                except ScannedPDFError as exc:
                    st.error(f"⚠️ {upload.name}: {exc}")
                except PDFIngestionError as exc:
                    st.error(f"❌ {upload.name}: {exc}")
                except Exception as exc:
                    st.error(f"❌ {upload.name}: indexing failed ({exc}).")

            if success_count > 0:
                st.success(f"✅ Successfully indexed {success_count} file(s)!")

    doc_items = list(st.session_state.documents.items())
    if doc_items:
        st.divider()
        st.subheader(f"Indexed Files ({len(doc_items)})")

        total_pages = sum(d["pages"] for _, d in doc_items)
        total_chunks = sum(d["chunks"] for _, d in doc_items)
        c1, c2 = st.columns(2)
        c1.metric("Total Pages", total_pages)
        c2.metric("Total Chunks", total_chunks)

        # Multi-document scope filter
        all_filenames = [d["filename"] for _, d in doc_items]
        selected_filenames = st.multiselect(
            "Active Scope (Search & Q&A)",
            options=all_filenames,
            default=all_filenames,
            help="Filter search and question answering to specific files, or keep all selected.",
        )
        if not selected_filenames:
            selected_filenames = all_filenames

        # Map selected filenames to document_ids
        target_doc_ids = [
            doc_id
            for doc_id, d in doc_items
            if d["filename"] in selected_filenames
        ]

        st.caption("Manage Individual Files:")
        for document_id, details in doc_items:
            with st.container():
                st.markdown(
                    f"📄 **{details['filename']}**  \n"
                    f"_{details['pages']} pages · {details['chunks']} chunks_"
                )
                if st.button(
                    "Delete",
                    key=f"del-{document_id}",
                    help=f"Remove {details['filename']}",
                    use_container_width=True,
                ):
                    delete_document(document_id)
                    st.rerun()

        if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
            clear_all_documents()
            st.rerun()
    else:
        target_doc_ids = None
        selected_filenames = []
        st.info("No documents indexed yet. Upload PDFs above to begin.")

    st.divider()
    st.caption("🔒 All embeddings and vectors are kept in-memory for this session.")

# Main Application Layout
st.markdown('<div class="main-header">📚 Search & Answer Across Multiple PDFs</div>', unsafe_allow_html=True)
st.caption("Search passages, discover cross-document insights, and ask questions with exact page-level citations.")

tab_qa, tab_search, tab_docs = st.tabs([
    "💬 Multi-File Question Answering",
    "🔍 Search Across Files",
    "📊 Document Overview",
])

# ==============================================================================
# TAB 1: QUESTION ANSWERING ACROSS MULTIPLE FILES
# ==============================================================================
with tab_qa:
    if doc_items:
        scope_text = ", ".join(selected_filenames)
        st.markdown(
            f'<div class="scope-banner">🌐 <strong>Active Scope:</strong> Answering across '
            f'<strong>{len(selected_filenames)} file(s)</strong>: <code>{scope_text}</code></div>',
            unsafe_allow_html=True,
        )

        if len(selected_filenames) >= 2:
            st.markdown("**Suggested cross-document prompts:**")
            chip1, chip2, chip3 = st.columns(3)
            if chip1.button("📊 Compare policies across all files", use_container_width=True):
                st.session_state.pending_question = "Compare the main policies, terms, and requirements across all documents."
                st.rerun()
            if chip2.button("📋 Summarize key points from each file", use_container_width=True):
                st.session_state.pending_question = "Summarize the key takeaways and purpose of each uploaded file."
                st.rerun()
            if chip3.button("⚖️ Note any conflicts or discrepancies", use_container_width=True):
                st.session_state.pending_question = "Are there any conflicting guidelines, differences, or discrepancies between these documents?"
                st.rerun()
    else:
        st.info("👈 Upload and index one or more PDFs in the sidebar to start asking questions.")

    # Render previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                doc_counts = message.get("doc_counts", {})
                count_str = " · ".join(f"{fn} ({c})" for fn, c in doc_counts.items())
                with st.expander(f"📚 Retrieved Evidence ({len(message['sources'])} chunks across {len(doc_counts)} file(s))"):
                    if count_str:
                        st.caption(f"**Coverage:** {count_str}")
                    for source in message["sources"]:
                        st.markdown(
                            f"<span class='badge badge-file'>{source['filename']}</span>"
                            f"<span class='badge badge-page'>Page {source['page']}</span>"
                            f"<span class='badge badge-score'>Similarity: {source['similarity']:.1%}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"> {source['preview']}")
                        st.markdown("---")

    # Handle incoming user query
    user_prompt = st.chat_input("Ask a question across the uploaded documents…")
    if st.session_state.pending_question:
        user_prompt = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            if not st.session_state.documents:
                answer = "Upload and process at least one PDF before asking a question."
                chunks = []
                st.warning(answer)
                sources = []
                doc_counts = {}
            else:
                with st.spinner("Retrieving evidence across documents and generating grounded answer…"):
                    chunks = st.session_state.rag_store.retrieve(
                        user_prompt,
                        top_k=8,
                        document_ids=target_doc_ids,
                        balanced_across_documents=True,
                    )
                    doc_counts = summarize_retrieval_by_document(chunks)

                    if not chunks:
                        answer = INSUFFICIENT_EVIDENCE
                    else:
                        api_key = os.getenv("DEEPSEEK_API_KEY", "")
                        model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
                        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

                        try:
                            answer = answer_question(
                                user_prompt,
                                chunks,
                                api_key=api_key,
                                model=model,
                                base_url=base_url,
                            )
                        except ValueError as exc:
                            answer = str(exc)
                        except Exception as exc:
                            answer = f"The answer service failed: {exc}"

                st.markdown(answer)

                sources = [
                    {
                        "filename": chunk.filename,
                        "page": chunk.page_number,
                        "citation": chunk.citation,
                        "similarity": chunk.similarity,
                        "preview": chunk.text[:500].replace("\n", " "),
                    }
                    for chunk in chunks
                ]

                if sources:
                    count_str = " · ".join(f"{fn} ({c})" for fn, c in doc_counts.items())
                    with st.expander(f"📚 Retrieved Evidence ({len(sources)} chunks across {len(doc_counts)} file(s))"):
                        if count_str:
                            st.caption(f"**Coverage:** {count_str}")
                        for source in sources:
                            st.markdown(
                                f"<span class='badge badge-file'>{source['filename']}</span>"
                                f"<span class='badge badge-page'>Page {source['page']}</span>"
                                f"<span class='badge badge-score'>Similarity: {source['similarity']:.1%}</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"> {source['preview']}")
                            st.markdown("---")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "doc_counts": doc_counts,
            }
        )

    if st.session_state.messages:
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages.clear()
            st.rerun()

# ==============================================================================
# TAB 2: MULTI-FILE SEARCH
# ==============================================================================
with tab_search:
    st.subheader("🔍 Search Across Multiple Files")
    st.caption("Search keywords, phrases, and semantic topics across all uploaded documents without calling the LLM.")

    if not doc_items:
        st.info("Upload and index PDFs first to enable multi-file search.")
    else:
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            search_query = st.text_input(
                "Search query",
                placeholder="e.g. remote work policy, confidentiality, payment terms, safety guidelines…",
                label_visibility="collapsed",
            )
        with sc2:
            search_button = st.button("Search", type="primary", use_container_width=True)

        with st.expander("Search Options & Filters", expanded=False):
            f1, f2, f3 = st.columns(3)
            with f1:
                search_top_k = st.slider("Max Results", min_value=3, max_value=25, value=8)
            with f2:
                min_similarity = st.slider("Min Similarity (%)", min_value=10, max_value=90, value=25)
            with f3:
                file_filter_options = ["All Selected Files"] + selected_filenames
                search_file_choice = st.selectbox("Search Target", options=file_filter_options)

        if search_query and (search_button or search_query):
            # Calculate max distance from min_similarity percentage
            max_dist = max(0.1, 1.0 - (min_similarity / 100.0))

            search_doc_ids = target_doc_ids
            if search_file_choice != "All Selected Files":
                search_doc_ids = [
                    doc_id for doc_id, d in doc_items if d["filename"] == search_file_choice
                ]

            with st.spinner(f"Searching across {len(search_doc_ids or doc_items)} files…"):
                search_results = st.session_state.rag_store.search(
                    query=search_query,
                    top_k=search_top_k,
                    max_distance=max_dist,
                    document_ids=search_doc_ids,
                )

            if not search_results:
                st.warning(
                    f"No passages found matching '{search_query}' with similarity >= {min_similarity}%. "
                    "Try lowering the minimum similarity threshold or using different keywords."
                )
            else:
                doc_breakdown = summarize_retrieval_by_document(search_results)
                breakdown_badges = " ".join(
                    f"<span class='badge badge-file'>{fn} ({cnt})</span>"
                    for fn, cnt in doc_breakdown.items()
                )
                st.markdown(
                    f"**Found {len(search_results)} match(es) across {len(doc_breakdown)} file(s):** {breakdown_badges}",
                    unsafe_allow_html=True,
                )
                st.markdown("")

                for idx, chunk in enumerate(search_results, start=1):
                    with st.container():
                        st.markdown(
                            f"<div class='search-card'>"
                            f"<strong>#{idx}</strong> &nbsp; "
                            f"<span class='badge badge-file'>{chunk.filename}</span>"
                            f"<span class='badge badge-page'>Page {chunk.page_number}</span>"
                            f"<span class='badge badge-score'>Match: {chunk.similarity:.1%}</span>"
                            f"<div style='margin-top: 0.6rem; line-height: 1.5;'>"
                            f"{highlight_query(chunk.text, search_query)}"
                            f"</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        btn_col1, _ = st.columns([2, 5])
                        with btn_col1:
                            if st.button(
                                f"💬 Ask Q&A about this excerpt",
                                key=f"ask_excerpt_{idx}",
                                help="Send this excerpt as context for a question in the Q&A tab",
                            ):
                                st.session_state.pending_question = (
                                    f"Based on {chunk.filename} page {chunk.page_number}, "
                                    f"explain: {chunk.text[:200]}..."
                                )
                                st.rerun()

# ==============================================================================
# TAB 3: DOCUMENT OVERVIEW & EXPLORER
# ==============================================================================
with tab_docs:
    st.subheader("📊 Document Overview & Explorer")
    st.caption("Inspect indexed documents, verify page boundaries, and audit extracted text chunks.")

    if not doc_items:
        st.info("No documents uploaded yet.")
    else:
        summary_rows = [
            {
                "Filename": d["filename"],
                "Pages": d["pages"],
                "Indexed Chunks": d["chunks"],
                "Document ID": doc_id[:16] + "…",
            }
            for doc_id, d in doc_items
        ]
        st.dataframe(summary_rows, use_container_width=True)

        st.divider()
        st.subheader("Inspect Document Chunks")
        doc_names = [d["filename"] for _, d in doc_items]
        inspect_doc_name = st.selectbox("Select document to inspect:", options=doc_names)

        selected_doc_entry = next((d for _, d in doc_items if d["filename"] == inspect_doc_name), None)
        if selected_doc_entry and "raw_chunks" in selected_doc_entry:
            chunks_to_view = selected_doc_entry["raw_chunks"]
            st.write(f"Showing **{len(chunks_to_view)} chunks** from **{inspect_doc_name}**:")
            for i, chunk in enumerate(chunks_to_view, start=1):
                with st.expander(f"Chunk #{i} — Page {chunk['page']} ({len(chunk['text'])} chars)"):
                    st.text(chunk["text"])
