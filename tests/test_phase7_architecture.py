from __future__ import annotations

import json
from pathlib import Path

from src.agents.clarifying_agent import ClarifyingAgent
from src.agents.inquiry_loop import InquiryLoop
from src.agents.orchestrator import AgentOrchestrator
from src.memory.patient_context import PatientContextManager
from src.rag.contracts import ClaimRecord
from src.rag.evidence import RelevanceGate, evidence_id_for_chunk, evidence_record_from_chunk, validate_claims
from src.rag.generator import CitationGenerator
from src.rag.pipeline import MedicalRAGPipeline
from src.guardrails.classifier import QueryClass, QueryClassification
from src.response import render_patient_response


def _chunk(chunk_id: str, score: float, title: str = "Headache") -> dict:
    return {
        "id": chunk_id,
        "rrf_score": score,
        "rerank_score": score,
        "text": "Evidence text",
        "metadata": {
            "chunk_id": chunk_id,
            "doc_title": title,
            "source": "NHS",
            "section": "Overview",
        },
    }


def test_evidence_ids_are_stable_and_claims_require_valid_evidence():
    chunk = _chunk("chunk-001", 0.9)
    assert evidence_id_for_chunk(chunk) == "chunk-001"
    evidence = [evidence_record_from_chunk(chunk, relevance_status="relevant")]

    valid, details = validate_claims(
        [
            ClaimRecord("c1", "Supported claim", ["chunk-001"]),
            ClaimRecord("c2", "Unsupported claim", ["missing-chunk"]),
        ],
        evidence,
    )
    assert [claim.claim_id for claim in valid] == ["c1"]
    assert details["invalid_claim_ids"] == ["c2"]


def test_relevance_gate_rejects_unrelated_candidates():
    selected, details = RelevanceGate().evaluate([_chunk("good", 0.8), _chunk("bad", 0.1)])
    assert [item["evidence_id"] for item in selected] == ["good"]
    assert details["passed"] is True
    assert details["rejected_ids"] == ["bad"]

    selected, details = RelevanceGate().evaluate([_chunk("bad", 0.1)])
    assert selected == []
    assert details["passed"] is False

    rrf_chunk = _chunk("rrf-good", 0.04)
    rrf_chunk["rerank_score_type"] = "rrf"
    rrf_chunk["text"] = "Headache evidence text"
    selected, _ = RelevanceGate().evaluate([rrf_chunk], query="headache")
    assert [item["evidence_id"] for item in selected] == ["rrf-good"]


class _Message:
    content = json.dumps(
        {
            "answer": "A supported answer.",
            "status": "answered",
            "claims": [{"claim_id": "c1", "text": "Supported claim", "evidence_ids": ["good"]}],
            "safety_actions": [],
        }
    )


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]


class _StructuredLLM:
    def complete(self, **kwargs):
        return _Response()


def test_generator_returns_structured_claims_and_validated_citations():
    result = CitationGenerator(llm_client=_StructuredLLM(), model="test/model").generate(
        "What is a headache?", [_chunk("good", 0.8)]
    )
    assert result["status"] == "answered"
    assert result["claims"][0]["evidence_ids"] == ["good"]
    assert result["citations"][0]["id"] == "good"
    assert "References" not in result["final_prompt"]


def test_patient_renderer_hides_internal_debug_fields():
    result = {
        "user_response": "Safe patient response.",
        "citations": [{"id": "secret-source"}],
        "final_prompt": "private prompt",
        "execution_time_seconds": 48.2,
        "retrieved_chunks": [{"text": "private context"}],
    }
    rendered = render_patient_response(result)
    assert rendered == "Safe patient response."
    assert "secret-source" not in rendered
    assert "private prompt" not in rendered
    assert "48.2" not in rendered


def test_patient_context_persists_and_hashes_user_reference(tmp_path: Path):
    path = tmp_path / "contexts.json"
    manager = PatientContextManager(path=path)
    key = manager.context_key("telegram:42")
    context = manager.create_or_update(key, {"symptoms": ["headache"], "sufficiency_score": 0.25})
    assert key != "telegram:42"
    assert manager.get(key).symptoms == ["headache"]
    assert json.loads(path.read_text(encoding="utf-8"))[key]["sufficiency_score"] == 0.25
    manager.clear(key)
    assert manager.get(key) is None


def test_clarifying_agent_distinguishes_general_and_personal_queries():
    agent = ClarifyingAgent()
    assert agent.analyze("What are common migraine symptoms?")["needs_inquiry"] is False
    assert agent.analyze("I have a headache")["needs_inquiry"] is True


def test_inquiry_loop_asks_one_question_then_reaches_sufficiency(tmp_path: Path):
    manager = PatientContextManager(path=tmp_path / "contexts.json")
    loop = InquiryLoop(context_manager=manager)
    key = manager.context_key("telegram:42")
    analysis = {"language": "en", "needs_inquiry": True}

    first = loop.process(key, "I have a headache", analysis)
    assert first["status"] == "clarification_needed"
    assert first["answer"]

    second = loop.process(key, "2 days", analysis)
    assert second["status"] == "clarification_needed"
    third = loop.process(key, "moderate", analysis)
    assert third["status"] == "clarification_needed"
    ready = loop.process(key, "no other symptoms", analysis)
    assert ready["status"] == "ready"
    assert ready["context"].sufficiency_score >= 0.7


def test_inquiry_loop_understands_colloquial_arabic_duration(tmp_path: Path):
    manager = PatientContextManager(path=tmp_path / "contexts.json")
    loop = InquiryLoop(context_manager=manager)
    key = manager.context_key("telegram:42")
    analysis = {"language": "ar", "needs_inquiry": True}

    first = loop.process(key, "عندي صداع في الرأس بقالو ٥ ايام", analysis)
    assert first["status"] == "clarification_needed"

    second = loop.process(key, "٥ ايام ويعتبر ثابت نفس الدرجة من الألم", analysis)
    assert second["context"].onset_or_duration
    assert "onset_or_duration" not in second["context"].missing_fields
    assert second["context"].trajectory
    assert second["answer"] != "منذ متى بدأ العرض؟ وهل يتحسن أم يزداد سوءاً؟"


def test_pipeline_skips_generation_when_relevance_gate_fails():
    pipeline = MedicalRAGPipeline(embedder=object(), qdrant_client=object())

    class Classifier:
        def classify(self, query):
            return QueryClassification(QueryClass.MEDICAL, 1.0, intent="medical_question")

    class Retriever:
        collection_name = "medical_kb"

        def retrieve(self, **kwargs):
            return [_chunk("unrelated", 0.1)]

    class Reranker:
        model_name = "test-reranker"

        def rerank(self, **kwargs):
            return kwargs["candidates"]

    class Generator:
        model = "test-generator"

        def generate(self, **kwargs):
            raise AssertionError("generation must not run after a failed relevance gate")

    pipeline.classifier = Classifier()
    pipeline.retriever = Retriever()
    pipeline.reranker = Reranker()
    pipeline.generator = Generator()

    result = pipeline.run("I have a headache", enable_query_rewrite=False)
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert result["relevance_gate"]["passed"] is False


def test_orchestrator_persists_inquiry_state_before_medical_agent(tmp_path: Path):
    class Classifier:
        def classify(self, query):
            return QueryClassification(QueryClass.MEDICAL, 1.0, intent="medical_question")

    class Documentation:
        pipeline = object()

        def __init__(self):
            self.calls = []

        def handle(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return {"answer": "medical answer", "status": "answered", "citations": []}

    documentation = Documentation()
    manager = PatientContextManager(path=tmp_path / "contexts.json")
    orchestrator = AgentOrchestrator(
        classifier=Classifier(),
        documentation_agent=documentation,
        context_manager=manager,
    )

    first = orchestrator.handle("I have a headache", user_ref="telegram:42")
    assert first["status"] == "clarification_needed"
    assert documentation.calls == []

    orchestrator.handle("2 days", user_ref="telegram:42")
    orchestrator.handle("moderate", user_ref="telegram:42")
    ready = orchestrator.handle("no other symptoms", user_ref="telegram:42")
    assert ready["agent"] == "documentation_agent"
    assert documentation.calls
    assert documentation.calls[-1][1]["patient_context"]["sufficiency_score"] >= 0.7


def test_active_inquiry_numeric_reply_cannot_become_emergency(tmp_path: Path):
    class Classifier:
        def classify(self, query):
            if query == "6":
                return QueryClassification(QueryClass.EMERGENCY, 0.99, intent="emergency", source="llm")
            return QueryClassification(QueryClass.MEDICAL, 1.0, intent="medical_question")

    class Documentation:
        pipeline = object()

        def handle(self, query, **kwargs):
            return {"answer": "medical answer", "status": "answered", "citations": []}

    manager = PatientContextManager(path=tmp_path / "contexts.json")
    orchestrator = AgentOrchestrator(
        classifier=Classifier(),
        documentation_agent=Documentation(),
        context_manager=manager,
    )

    first = orchestrator.handle("I have a headache", user_ref="telegram:99")
    assert first["status"] == "clarification_needed"
    second = orchestrator.handle("6", user_ref="telegram:99")
    assert second["route"] != "emergency"


def test_reset_session_removes_active_context(tmp_path: Path):
    manager = PatientContextManager(path=tmp_path / "contexts.json")
    manager.create_or_update(manager.context_key("telegram:99"), {"status": "collecting"})
    manager.reset_for_user("telegram:99")
    assert manager.get(manager.context_key("telegram:99")) is None


def test_ambiguous_arabic_message_gets_arabic_missing_symptom_prompt(tmp_path: Path):
    class Classifier:
        def classify(self, query):
            return QueryClassification(QueryClass.MEDICAL, 0.1, intent="medical_question")

    manager = PatientContextManager(path=tmp_path / "contexts.json")
    orchestrator = AgentOrchestrator(classifier=Classifier(), context_manager=manager)
    result = orchestrator.handle("متوسط", user_ref="telegram:100")
    assert "العرض" in result["user_response"]
    assert "route" not in result["user_response"]
