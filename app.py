"""Streamlit user interface for page-cited PDF question answering."""

from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import importlib
import rag
importlib.reload(rag)

from ingest import PDFIngestionError, ScannedPDFError, ingest_pdf
from rag import INSUFFICIENT_EVIDENCE, RAGStore, answer_question


load_dotenv()
st.set_page_config(page_title="PDF Q&A with RAG", page_icon="📚", layout="wide")


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
        st.session_state.rag_store = RAGStore(
            load_embedding_model(embedding_name), st.session_state.session_id
        )


def delete_document(document_id: str) -> None:
    st.session_state.rag_store.delete_document(document_id)
    st.session_state.documents.pop(document_id, None)


initialize_session()

st.title("📚 PDF Question Answering")
st.caption("Upload text-based PDFs, then ask questions answered only from retrieved evidence.")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs only; maximum 25 MB per file.",
    )
    if st.button("Process selected PDFs", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Select at least one PDF first.")
        else:
            for upload in uploads:
                try:
                    document = ingest_pdf(upload.name, upload.getvalue())
                    if document.document_id in st.session_state.documents:
                        st.info(f"{document.filename} is already indexed.")
                        continue
                    with st.spinner(f"Embedding {document.filename}…"):
                        chunk_count = st.session_state.rag_store.add_document(document)
                    st.session_state.documents[document.document_id] = {
                        "filename": document.filename,
                        "pages": document.page_count,
                        "chunks": chunk_count,
                    }
                    st.success(f"Indexed {document.filename} ({chunk_count} chunks).")
                except ScannedPDFError as exc:
                    st.error(f"{upload.name}: {exc}")
                except PDFIngestionError as exc:
                    st.error(f"{upload.name}: {exc}")
                except Exception as exc:
                    st.error(f"{upload.name}: indexing failed ({exc}).")

    if st.session_state.documents:
        st.divider()
        for document_id, details in list(st.session_state.documents.items()):
            st.markdown(f"**{details['filename']}**  \n{details['pages']} pages · {details['chunks']} chunks")
            if st.button(
                "Delete",
                key=f"delete-{document_id}",
                help=f"Remove {details['filename']} from this session",
                use_container_width=True,
            ):
                delete_document(document_id)
                st.rerun()
    else:
        st.info("No documents indexed in this session.")

    st.divider()
    st.caption("Documents and vectors are isolated to this browser session and kept in memory.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved evidence"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['citation']}** · similarity {source['similarity']:.2f}\n\n"
                        f"> {source['preview']}"
                    )

question = st.chat_input("Ask a question about the uploaded PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.documents:
            answer = "Upload and process at least one PDF before asking a question."
            chunks = []
            st.warning(answer)
        else:
            with st.spinner("Retrieving evidence and generating an answer…"):
                chunks = st.session_state.rag_store.retrieve(question)
                if not chunks:
                    answer = INSUFFICIENT_EVIDENCE
                else:
                    provider = os.getenv("LLM_PROVIDER", "").lower()
                    if not provider:
                        if os.getenv("DEEPSEEK_API_KEY"):
                            provider = "deepseek"
                        elif os.getenv("OPENAI_API_KEY"):
                            provider = "openai"
                        else:
                            provider = "ollama"

                    if provider == "deepseek":
                        api_key = os.getenv("DEEPSEEK_API_KEY", "")
                        model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
                        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
                    elif provider == "openai":
                        api_key = os.getenv("OPENAI_API_KEY", "")
                        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                        base_url = None
                    else:
                        api_key = ""
                        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                        base_url = None

                    try:
                        answer = answer_question(
                            question,
                            chunks,
                            provider=provider,
                            api_key=api_key,
                            model=model,
                            base_url=base_url,
                            ollama_base_url=os.getenv(
                                "OLLAMA_BASE_URL", "http://localhost:11434"
                            ),
                        )
                    except ValueError as exc:
                        answer = str(exc)
                    except Exception as exc:
                        answer = f"The answer service failed: {exc}"
            st.markdown(answer)

        sources = [
            {
                "citation": chunk.citation,
                "similarity": max(0.0, 1.0 - chunk.distance),
                "preview": chunk.text[:500].replace("\n", " "),
            }
            for chunk in chunks
        ]
        if sources:
            with st.expander("Retrieved evidence"):
                for source in sources:
                    st.markdown(
                        f"**{source['citation']}** · similarity {source['similarity']:.2f}\n\n"
                        f"> {source['preview']}"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
