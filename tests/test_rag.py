from types import SimpleNamespace

import numpy as np

from ingest import IngestedDocument, TextChunk
from rag import (
    INSUFFICIENT_EVIDENCE,
    RAGStore,
    RetrievedChunk,
    answer_question,
    summarize_retrieval_by_document,
)


class FakeEmbeddingModel:
    """Small deterministic embedding model that avoids network/model downloads."""

    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        vectors = []
        for sentence in sentences:
            lower = sentence.lower()
            vector = np.array(
                [lower.count("alpha"), lower.count("beta"), lower.count("policy"), 0.1],
                dtype=float,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


def make_document() -> IngestedDocument:
    chunks = (
        TextChunk("doc:0", "doc", "guide.pdf", 1, 0, "alpha policy allows remote work"),
        TextChunk("doc:1", "doc", "guide.pdf", 2, 1, "beta procedure requires approval"),
    )
    return IngestedDocument("doc", "guide.pdf", 2, chunks)


def test_add_retrieve_and_delete_document():
    store = RAGStore(FakeEmbeddingModel(), "test-session")
    assert store.add_document(make_document()) == 2

    results = store.retrieve("What is the alpha policy?", top_k=1, max_distance=1.0)

    assert len(results) == 1
    assert results[0].filename == "guide.pdf"
    assert results[0].page_number == 1
    store.delete_document("doc")
    assert store.retrieve("alpha") == []


def test_distance_threshold_can_reject_weak_matches():
    store = RAGStore(FakeEmbeddingModel(), "threshold-session")
    store.add_document(make_document())
    assert store.retrieve("unrelated question", max_distance=0.01) == []


def test_answer_returns_fallback_without_calling_api():
    assert answer_question("Unknown?", [], api_key="test") == INSUFFICIENT_EVIDENCE


def test_answer_requires_api_key_when_no_client():
    import pytest
    chunks = [RetrievedChunk("Remote work is allowed.", "guide.pdf", 1, "doc", 0.1)]
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        answer_question("Is remote work allowed?", chunks, api_key="")


def test_answer_calls_deepseek_chat_api():
    captured = {}

    class FakeChoices:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[FakeChoices("Remote work is allowed [guide.pdf, p. 1].")])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions()))
    chunks = [RetrievedChunk("Remote work is allowed.", "guide.pdf", 1, "doc", 0.1)]

    answer = answer_question(
        "Is remote work allowed?",
        chunks,
        api_key="test-key",
        model="deepseek-v4-flash",
        client=fake_client,
    )

    assert answer == "Remote work is allowed [guide.pdf, p. 1]."
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["temperature"] == 0
    assert "[guide.pdf, p. 1]" in captured["messages"][1]["content"]
    assert "only the evidence" in captured["messages"][0]["content"]


def make_second_document() -> IngestedDocument:
    chunks = (
        TextChunk("doc2:0", "doc2", "handbook.pdf", 1, 0, "alpha policy requires security training"),
        TextChunk("doc2:1", "doc2", "handbook.pdf", 3, 1, "general information without terms"),
    )
    return IngestedDocument("doc2", "handbook.pdf", 3, chunks)


def test_search_across_multiple_documents():
    store = RAGStore(FakeEmbeddingModel(), "multi-search-session")
    store.add_document(make_document())
    store.add_document(make_second_document())
    assert store.document_count == 2

    results = store.search("alpha policy", top_k=5, max_distance=1.0)
    filenames = {r.filename for r in results}

    assert "guide.pdf" in filenames
    assert "handbook.pdf" in filenames
    counts = summarize_retrieval_by_document(results)
    assert counts.get("guide.pdf", 0) >= 1
    assert counts.get("handbook.pdf", 0) >= 1


def test_filter_retrieval_by_document_id():
    store = RAGStore(FakeEmbeddingModel(), "filter-session")
    store.add_document(make_document())
    store.add_document(make_second_document())

    # Retrieve only from doc2 (handbook.pdf)
    results = store.retrieve("alpha policy", top_k=5, max_distance=1.0, document_ids=["doc2"])
    assert len(results) >= 1
    assert all(r.filename == "handbook.pdf" for r in results)

    # Empty document_ids list returns empty list
    assert store.retrieve("alpha policy", document_ids=[]) == []


def test_balanced_multi_document_retrieval():
    store = RAGStore(FakeEmbeddingModel(), "balanced-session")
    store.add_document(make_document())
    store.add_document(make_second_document())

    results = store.retrieve(
        "alpha policy",
        top_k=2,
        max_distance=1.0,
        balanced_across_documents=True,
    )
    # Balanced retrieval ensures both documents get representation when top_k >= 2
    filenames = {r.filename for r in results}
    assert "guide.pdf" in filenames
    assert "handbook.pdf" in filenames


def test_answer_synthesizes_across_multiple_files():
    captured = {}

    class FakeChoices:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    FakeChoices(
                        "Remote work is permitted [guide.pdf, p. 1], but security training is mandatory [handbook.pdf, p. 2]."
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions()))
    chunks = [
        RetrievedChunk("Remote work is permitted.", "guide.pdf", 1, "doc", 0.1),
        RetrievedChunk("Security training is mandatory.", "handbook.pdf", 2, "doc2", 0.12),
    ]

    answer = answer_question(
        "Compare remote work and security rules across documents",
        chunks,
        api_key="test-key",
        client=fake_client,
    )

    assert "[guide.pdf, p. 1]" in answer
    assert "[handbook.pdf, p. 2]" in answer
    # Both sources should be present in evidence prompt
    prompt_evidence = captured["messages"][1]["content"]
    assert "[guide.pdf, p. 1]" in prompt_evidence
    assert "[handbook.pdf, p. 2]" in prompt_evidence
    # Grounding instructions prompt
    system_prompt = captured["messages"][0]["content"]
    assert "across multiple files" in system_prompt


def test_rag_store_recovers_from_corrupt_ephemeral_system_cache():
    import chromadb
    # Intentionally stop an ephemeral client to simulate Chroma's deleted bindings state
    c = chromadb.EphemeralClient()
    c._system.stop()

    # RAGStore should safely self-heal, clearing cache or falling back
    store = RAGStore(FakeEmbeddingModel(), "recovery-session")
    doc = make_document()
    added = store.add_document(doc)
    assert added == 2
    results = store.search("remote work policy", top_k=2)
    assert len(results) >= 1



