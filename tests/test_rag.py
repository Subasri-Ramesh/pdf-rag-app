from types import SimpleNamespace

import numpy as np

from ingest import IngestedDocument, TextChunk
from rag import INSUFFICIENT_EVIDENCE, RAGStore, RetrievedChunk, answer_question


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
    assert (
        answer_question("Unknown?", [], provider="ollama", model="test")
        == INSUFFICIENT_EVIDENCE
    )


def test_answer_calls_responses_api_with_citations_and_no_storage():
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="Remote work is allowed [guide.pdf, p. 1].")

    fake_client = SimpleNamespace(responses=FakeResponses())
    chunks = [RetrievedChunk("Remote work is allowed.", "guide.pdf", 1, "doc", 0.1)]

    answer = answer_question(
        "Is remote work allowed?",
        chunks,
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=fake_client,
    )

    assert answer == "Remote work is allowed [guide.pdf, p. 1]."
    assert captured["model"] == "test-model"
    assert captured["store"] is False
    assert "[guide.pdf, p. 1]" in captured["input"]
    assert "only the evidence" in captured["instructions"]


def test_answer_calls_local_ollama_chat_api():
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "Allowed [guide.pdf, p. 1]."}}

    class FakeHTTPClient:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

    chunks = [RetrievedChunk("Remote work is allowed.", "guide.pdf", 1, "doc", 0.1)]
    answer = answer_question(
        "Is remote work allowed?",
        chunks,
        provider="ollama",
        model="llama3.2:3b",
        client=FakeHTTPClient(),
    )

    assert answer == "Allowed [guide.pdf, p. 1]."
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["stream"] is False
    assert captured["json"]["model"] == "llama3.2:3b"
    assert "[guide.pdf, p. 1]" in captured["json"]["messages"][1]["content"]


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
        provider="deepseek",
        api_key="test-key",
        model="deepseek-v4-flash",
        client=fake_client,
    )

    assert answer == "Remote work is allowed [guide.pdf, p. 1]."
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["temperature"] == 0
    assert "[guide.pdf, p. 1]" in captured["messages"][1]["content"]
    assert "only the evidence" in captured["messages"][0]["content"]

