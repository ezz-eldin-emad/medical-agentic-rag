from __future__ import annotations

import json
from pathlib import Path

from src.agents.clinic_agent import ClinicAgent
from src.agents.orchestrator import AgentOrchestrator
from src.guardrails.classifier import QueryClass, QueryClassification, QueryClassifier

CLINIC_FIXTURE = Path(__file__).parent / "fixtures" / "clinic_info.json"


class FakeMessage:
    content = '{"query_class":"clinic_query","intent":"availability","confidence":0.94,"reason":"appointment wording","entities":{"specialty":"صدرية","weekday":"السبت"}}'


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeLLM:
    def complete(self, **kwargs):
        return FakeResponse()


def test_llm_classifier_parses_structured_routing_result():
    classifier = QueryClassifier(llm_client=FakeLLM(), use_llm=True)
    result = classifier.classify("هل يوجد موعد للصدرية يوم السبت؟")
    assert result.query_class is QueryClass.CLINIC
    assert result.intent == "availability"
    assert result.source == "llm"
    assert result.entities["specialty"] == "صدرية"


def test_clinic_agent_reads_availability_without_llm(tmp_path: Path):
    bookings = tmp_path / "bookings.json"
    agent = ClinicAgent(clinic_path=CLINIC_FIXTURE, bookings_path=bookings)
    result = agent.lookup(intent="availability", entities={"doctor_id": "DR001", "weekday": "السبت"})
    assert result["route"] == "clinic_query"
    assert "د. أحمد السيد" in result["answer"]
    assert not bookings.exists()


def test_clinic_agent_returns_structured_prices_without_llm(tmp_path: Path):
    result = ClinicAgent(clinic_path=CLINIC_FIXTURE, bookings_path=tmp_path / "bookings.json").lookup(intent="clinic_info")
    assert "كشف عظام" in result["answer"]
    assert "320 EGP" in result["answer"]


def test_clinic_booking_confirm_cancel_and_persists(tmp_path: Path):
    bookings = tmp_path / "bookings.json"
    agent = ClinicAgent(clinic_path=CLINIC_FIXTURE, bookings_path=bookings)
    entities = {"doctor_id": "DR001", "date": "2026-08-29", "time": "10:30"}

    created = agent.book(entities, user_ref="telegram:42")
    booking_id = created["booking"]["booking_id"]
    assert created["status"] == "pending"
    assert json.loads(bookings.read_text(encoding="utf-8"))[0]["booking_id"] == booking_id

    confirmed = agent.confirm(booking_id, user_ref="telegram:42")
    assert confirmed["status"] == "confirmed"
    cancelled = agent.cancel(booking_id, user_ref="telegram:42")
    assert cancelled["status"] == "cancelled"


def test_clinic_booking_rejects_schedule_conflicts(tmp_path: Path):
    agent = ClinicAgent(clinic_path=CLINIC_FIXTURE, bookings_path=tmp_path / "bookings.json")
    entities = {"doctor_id": "DR001", "date": "2026-08-29", "time": "10:30"}
    assert agent.book(entities, user_ref="telegram:1")["status"] == "pending"
    assert agent.book(entities, user_ref="telegram:2")["status"] == "conflict"


class FakeSafety:
    def handle(self, **kwargs):
        return {"answer": "emergency", "citations": []}


class FakeClinic:
    def handle(self, **kwargs):
        return {"answer": "clinic", "citations": []}


class FakeDocumentation:
    pipeline = object()

    def handle(self, query, **kwargs):
        return {"answer": "medical", "citations": []}


def test_orchestrator_routes_each_class_to_one_agent():
    class FakeClassifier:
        def __init__(self, classification):
            self.classification = classification

        def classify(self, query):
            return self.classification

    for query_class, expected_agent, expected_answer in (
        (QueryClass.EMERGENCY, "safety_agent", "emergency"),
        (QueryClass.CLINIC, "clinic_agent", "clinic"),
        (QueryClass.MEDICAL, "documentation_agent", "medical"),
    ):
        orchestrator = AgentOrchestrator(
            classifier=FakeClassifier(QueryClassification(query_class, 1.0)),
            safety_agent=FakeSafety(),
            clinic_agent=FakeClinic(),
            documentation_agent=FakeDocumentation(),
        )
        result = orchestrator.handle("query", user_ref="telegram:42")
        assert result["agent"] == expected_agent
        assert result["answer"] == expected_answer
