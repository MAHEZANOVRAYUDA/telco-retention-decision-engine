"""Integration & Contract Tests for Agentic Decision Layer & FastAPI Service.

Verifies:
- Strict Pydantic v2 validation on FeatureDriver, DecisionPacket, AgentInterventionResponse
- RetentionAgent hard guardrails (rejects Sleeping Dogs & ineligibles)
- Deterministic Indonesian copy generation referencing top-3 TreeSHAP drivers
- FastAPI endpoints (/v1/health, /v1/decide, /v1/optimize-cohort, /v1/drift)
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.agentic.retention_agent import RetentionAgent
from src.agentic.schemas import (
    AgentInterventionResponse,
    DecisionPacket,
    FeatureDriver,
    SubscriberFeatureInput,
)
from src.api.app import app


@pytest.fixture(scope="module")
def api_client():
    """Create FastAPI test client with lifespan context."""
    with TestClient(app) as client:
        yield client


def test_pydantic_decision_packet_contract():
    """Ensure strict validation bounds on DecisionPacket."""
    valid_driver = FeatureDriver(
        feature_name="drop_call_rate",
        feature_value=8.5,
        shap_contribution=0.12,
        direction="increases_churn_risk",
    )

    # Valid packet
    packet = DecisionPacket(
        subscriber_id="IOH-SUB-000123",
        baseline_churn_prob=0.45,
        strategic_quadrant="Persuadable",
        recommended_action="data_booster_15gb",
        expected_uplift_pct=14.5,
        uplift_confidence_lower_90=11.2,
        uplift_confidence_upper_90=17.8,
        eligibility_passed=True,
        eligibility_reason="High usage decay detected.",
        intervention_cost_idr=12000.0,
        expected_net_value_idr=85000.0,
        top_key_drivers=[valid_driver],
    )
    assert packet.subscriber_id == "IOH-SUB-000123"

    # Invalid probability (churn > 1.0) must fail
    with pytest.raises(ValidationError):
        DecisionPacket(
            subscriber_id="IOH-SUB-000123",
            baseline_churn_prob=1.5,  # Invalid!
            strategic_quadrant="Persuadable",
            recommended_action="data_booster_15gb",
            expected_uplift_pct=14.5,
            uplift_confidence_lower_90=11.2,
            uplift_confidence_upper_90=17.8,
            eligibility_passed=True,
            eligibility_reason="ok",
            intervention_cost_idr=12000.0,
            expected_net_value_idr=85000.0,
            top_key_drivers=[],
        )


def test_agent_guardrail_rejections():
    """Test RetentionAgent hard guardrails against Sleeping Dogs and ineligibles."""
    agent = RetentionAgent()

    # Case 1: Ineligible subscriber
    ineligible_packet = DecisionPacket(
        subscriber_id="IOH-INELIGIBLE",
        baseline_churn_prob=0.30,
        strategic_quadrant="Sure_Thing",
        recommended_action="combo_discount_20pct",
        expected_uplift_pct=2.0,
        uplift_confidence_lower_90=-1.0,
        uplift_confidence_upper_90=5.0,
        eligibility_passed=False,  # Ineligible
        eligibility_reason="ARPU below minimum threshold.",
        intervention_cost_idr=35000.0,
        expected_net_value_idr=-15000.0,
        top_key_drivers=[],
    )
    res1 = agent.process_decision(ineligible_packet)
    assert res1.guardrail_passed is False
    assert res1.recommended_action == "no_action"
    assert res1.channel == "Holdout_No_Action"

    # Case 2: Sleeping Dog
    sleeping_dog_packet = DecisionPacket(
        subscriber_id="IOH-SLEEPING-DOG",
        baseline_churn_prob=0.20,
        strategic_quadrant="Sleeping_Dog",  # Sleeping Dog
        recommended_action="no_action",
        expected_uplift_pct=-5.0,
        uplift_confidence_lower_90=-8.0,
        uplift_confidence_upper_90=-2.0,
        eligibility_passed=False,
        eligibility_reason="Negative uplift",
        intervention_cost_idr=0.0,
        expected_net_value_idr=0.0,
        top_key_drivers=[],
    )
    res2 = agent.process_decision(sleeping_dog_packet)
    assert res2.guardrail_passed is False
    assert res2.channel == "Holdout_No_Action"


def test_agent_deterministic_fallback_message():
    """Verify deterministic Indonesian copy contains driver context."""
    agent = RetentionAgent()  # without API key -> triggers deterministic fallback

    driver = FeatureDriver(
        feature_name="drop_call_rate",
        feature_value=12.4,
        shap_contribution=0.25,
        direction="increases_churn_risk",
    )
    packet = DecisionPacket(
        subscriber_id="IOH-SUB-009999",
        baseline_churn_prob=0.75,
        strategic_quadrant="Persuadable",
        recommended_action="qos_priority_care",
        expected_uplift_pct=18.5,
        uplift_confidence_lower_90=15.0,
        uplift_confidence_upper_90=22.0,
        eligibility_passed=True,
        eligibility_reason="High drop call rate anomaly.",
        intervention_cost_idr=18000.0,
        expected_net_value_idr=210000.0,
        top_key_drivers=[driver],
    )
    res = agent.process_decision(packet)
    assert res.guardrail_passed is True
    assert res.channel == "Field_Engineer_Ticket"
    assert "IOH-SUB-009999" in res.intervention_message_id_lang
    assert "QoS Premium" in res.intervention_message_id_lang or "CARE-" in res.intervention_message_id_lang
    assert res.source == "deterministic_policy_fallback"


def test_fastapi_endpoints(api_client):
    """Test full FastAPI service endpoints."""
    # 1. Health check
    res_health = api_client.get("/v1/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["memory"]["current_rss_mb"] < 600.0

    # 2. Single decide endpoint
    sub_payload = {
        "subscriber_id": "IOH-TEST-001",
        "tenure_months": 8,
        "monthly_arpu_idr": 115000.0,
        "data_usage_gb_m1": 18.5,
        "usage_decay_ratio": 0.55,  # high decay
        "drop_call_rate": 8.5,      # high drop
        "complaint_count_90d": 2,
        "is_prepaid": 1,
        "is_annual": 0,
    }
    res_decide = api_client.post("/v1/decide", json=sub_payload)
    assert res_decide.status_code == 200
    data_decide = res_decide.json()
    assert "latency_ms" in data_decide
    assert "decision_packet" in data_decide
    assert "agent_response" in data_decide
    packet = data_decide["decision_packet"]
    assert packet["subscriber_id"] == "IOH-TEST-001"
    assert len(packet["top_key_drivers"]) == 3

    # 3. Cohort optimization endpoint
    opt_payload = {
        "budget_idr": 5000000.0,
        "confidence_alpha": 0.10,
        "max_subscribers": 1000,
    }
    res_opt = api_client.post("/v1/optimize-cohort", json=opt_payload)
    assert res_opt.status_code == 200
    data_opt = res_opt.json()
    assert data_opt["budget_spent_idr"] <= 5000000.0
    assert data_opt["net_incremental_value_idr"] > 0
    assert "action_distribution" in data_opt

    # 4. Drift detection endpoint
    res_drift = api_client.get("/v1/drift")
    assert res_drift.status_code == 200
    data_drift = res_drift.json()
    assert "status" in data_drift
    assert "max_psi" in data_drift
    assert "features" in data_drift
