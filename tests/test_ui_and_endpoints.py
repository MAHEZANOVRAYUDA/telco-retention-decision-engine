"""Integration & Aesthetic Validation Tests for Dominant-White Swiss Editorial UI.

Verifies:
- GET / serves Swiss Editorial Decision Simulator HTML (200 OK)
- Dominant-White design guidelines (#FFFFFF, #F8FAFC, 1px solid #E2E8F0 hairline border)
- Precision typography (Inter, JetBrains Mono tabular numerals)
- Zero AI-slop (no neon gradients, no generic purple buttons)
- Live backend endpoint connectivity
"""

import re
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_serve_simulator_html(client):
    """GET / must return valid 200 HTML console."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text

    # Title verification
    assert "GARUDA-NBA — Stackelberg Retention & Causal Uplift Decision Console" in html

    # Tab navigation elements
    assert "tabBtnCohort" in html
    assert "tabBtnSubscriber" in html
    assert "tabBtnDrift" in html


def test_swiss_editorial_styling_contracts(client):
    """Strict inspection of Dominant-White Swiss Editorial aesthetic tokens."""
    res = client.get("/")
    html = res.text

    # 1. Backgrounds & Surfaces: Must enforce #FFFFFF (canvas) & #F8FAFC (subtle)
    assert "#FFFFFF" in html or "#ffffff" in html or "bg-white" in html
    assert "#F8FAFC" in html or "bg-slate-50" in html

    # 2. Hairline borders: Must enforce #E2E8F0 (border-slate-200)
    assert "#E2E8F0" in html or "border-slate-200" in html

    # 3. Precision Typography: Inter & JetBrains Mono with tabular numbers
    assert "Inter" in html
    assert "JetBrains Mono" in html
    assert "tabular-nums" in html

    # 4. Zero AI-Slop Enforcement: Strictly forbid gaudy neon/purple gradient cards
    assert "linear-gradient(to right, #8b5cf6" not in html
    assert "from-purple-600" not in html
    assert "to-indigo-600" not in html
    assert "shadow-neon" not in html


def test_ui_backend_endpoints_functional(client):
    """Test full pipeline interaction from UI perspectives."""
    # Health endpoint
    h_res = client.get("/v1/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"

    # Cohort optimization endpoint
    c_res = client.post("/v1/optimize-cohort", json={"budget_idr": 12000000.0, "confidence_alpha": 0.10})
    assert c_res.status_code == 200
    c_data = c_res.json()
    assert c_data["budget_spent_idr"] <= 12000000.0
    assert c_data["net_incremental_value_idr"] > 0

    # Decide single subscriber
    d_res = client.post(
        "/v1/decide",
        json={
            "subscriber_id": "IOH-SUB-015004",
            "tenure_months": 14,
            "monthly_arpu_idr": 145000.0,
            "data_usage_gb_m1": 16.5,
            "usage_decay_ratio": 0.62,
            "drop_call_rate": 8.4,
            "complaint_count_90d": 2,
            "is_prepaid": 1,
            "is_annual": 0,
        },
    )
    assert d_res.status_code == 200
    d_data = d_res.json()
    assert d_data["decision_packet"]["subscriber_id"] == "IOH-SUB-015004"
    assert d_data["agent_response"]["guardrail_passed"] is True

