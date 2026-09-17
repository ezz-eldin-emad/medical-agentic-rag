from src.agents.tracing import NoopTracer


def test_noop_tracer_preserves_span_state():
    tracer = NoopTracer()
    with tracer.span("agent_orchestrator", {"query_length": 4}) as state:
        state["classification"] = "clinic_query"
    assert state["name"] == "agent_orchestrator"
    assert state["classification"] == "clinic_query"
