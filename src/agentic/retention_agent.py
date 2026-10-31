"""Autonomous Retention Copilot Agent.

Consumes validated DecisionPackets, verifies strict guardrails
(rejecting Sleeping Dogs and ineligible promotions), and synthesizes
personalized Indonesian customer interventions using Groq (llama-3.3-70b-versatile)
with a deterministic, zero-dependency policy fallback for 100% offline reliability.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

from src.agentic.schemas import AgentInterventionResponse, DecisionPacket
from src.config import GROQ_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.retention_agent")


class RetentionAgent:
    """Enterprise retention copilot with deterministic guardrails and Groq LLM integration."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.model = GROQ_MODEL
        self.groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def process_decision(self, packet: DecisionPacket) -> AgentInterventionResponse:
        """Process DecisionPacket through guardrails and generate intervention payload."""
        # Hard Guardrail 1: Check eligibility
        if not packet.eligibility_passed:
            logger.info(f"Guardrail rejection for {packet.subscriber_id}: eligibility failed ({packet.eligibility_reason})")
            return AgentInterventionResponse(
                subscriber_id=packet.subscriber_id,
                channel="Holdout_No_Action",
                guardrail_passed=False,
                guardrail_rejection_reason=f"Eligibility check failed: {packet.eligibility_reason}",
                intervention_message_id_lang="Pelanggan tidak memenuhi syarat untuk intervensi aktif saat ini.",
                technical_summary=f"Excluded from campaign: {packet.eligibility_reason}",
                recommended_action="no_action",
                intervention_cost_idr=0.0,
                expected_net_value_idr=0.0,
                source="deterministic_policy_fallback",
            )

        # Hard Guardrail 2: Reject Sleeping Dogs or no_action recommendations
        if packet.strategic_quadrant == "Sleeping_Dog" or packet.recommended_action == "no_action":
            logger.info(f"Guardrail rejection for {packet.subscriber_id}: Sleeping Dog or holdout quadrant ({packet.strategic_quadrant})")
            return AgentInterventionResponse(
                subscriber_id=packet.subscriber_id,
                channel="Holdout_No_Action",
                guardrail_passed=False,
                guardrail_rejection_reason=f"Restricted quadrant: {packet.strategic_quadrant} to prevent negative churn uplift",
                intervention_message_id_lang="Intervensi dibatalkan secara otomatis guna melindungi pendapatan dan retensi organik.",
                technical_summary=f"Guardrail active: {packet.strategic_quadrant} detected. Outbound message strictly prohibited.",
                recommended_action="no_action",
                intervention_cost_idr=0.0,
                expected_net_value_idr=0.0,
                source="deterministic_policy_fallback",
            )

        # Map channel
        channel = self._resolve_channel(packet.recommended_action)

        # Attempt Groq API if key is present
        if self.api_key:
            llm_result = self._generate_with_groq(packet, channel)
            if llm_result:
                return llm_result

        # Fallback to Deterministic Policy Template
        return self._generate_deterministic(packet, channel)

    def _resolve_channel(self, action: str) -> str:
        """Map recommended action to high-converting touchpoint."""
        if action == "qos_priority_care":
            return "Field_Engineer_Ticket"
        elif action == "data_booster_15gb":
            return "MyIOH_Push"
        elif action == "combo_discount_20pct":
            return "WhatsApp"
        return "Holdout_No_Action"

    def _generate_deterministic(self, packet: DecisionPacket, channel: str) -> AgentInterventionResponse:
        """Deterministic policy generation (0 ms latency, 100% offline reproducible)."""
        action = packet.recommended_action
        sub_id = packet.subscriber_id

        # Format key drivers explanation
        driver_details = []
        for d in packet.top_key_drivers[:2]:
            driver_details.append(f"{d.feature_name.replace('_', ' ')}: {d.feature_value}")
        driver_str = ", ".join(driver_details) if driver_details else "telemetri penggunaan"

        if action == "data_booster_15gb":
            msg = (
                f"Halo Pelanggan Setia IOH ({sub_id}), kami mengamati kuota data Anda hampir habis berdasarkan "
                f"analisis pola konsumsi Anda ({driver_str}). Spesial hari ini, kami mengaktifkan Kuota Tambahan "
                f"Data Booster 15 GB (7 Hari) untuk menjaga konektivitas digital Anda tetap lancar!"
            )
            tech = f"Injected 15GB 7D high-priority data slice based on usage decay ({packet.expected_uplift_pct:.1f}% expected uplift)."
        elif action == "combo_discount_20pct":
            msg = (
                f"Yth. Pelanggan Prioritas IOH ({sub_id}), terima kasih atas loyalitas Anda bersama kami. "
                f"Sebagai bentuk apresiasi atas hubungan kemitraan kita ({driver_str}), kami memberikan Diskon "
                f"Spesial 20% untuk paket langganan 3 bulan ke depan melalui aplikasi MyIOH."
            )
            tech = f"Applied 20% renewal contract discount subsidy (Net EV: Rp {packet.expected_net_value_idr:,.0f})."
        elif action == "qos_priority_care":
            msg = (
                f"Pemberitahuan Kualitas Layanan IOH ({sub_id}): Tim Network Operations kami mendeteksi fluktuasi "
                f"sinyal di BTS area Anda ({driver_str}). Prioritas koneksi nomor Anda telah ditingkatkan ke jalur "
                f"QoS Premium dan tiket perbaikan #CARE-{sub_id[-4:]} telah otomatis diterbitkan untuk pemantauan 24 jam."
            )
            tech = f"Opened auto-care NOC priority ticket and elevated radio bearer slice (CATE Uplift: +{packet.expected_uplift_pct:.1f}%)."
        else:
            msg = f"Halo Pelanggan IOH ({sub_id}), terima kasih telah mempercayakan layanan komunikasi Anda kepada kami."
            tech = "Baseline monitoring."

        return AgentInterventionResponse(
            subscriber_id=sub_id,
            channel=channel,
            guardrail_passed=True,
            guardrail_rejection_reason=None,
            intervention_message_id_lang=msg,
            technical_summary=tech,
            recommended_action=action,
            intervention_cost_idr=packet.intervention_cost_idr,
            expected_net_value_idr=packet.expected_net_value_idr,
            source="deterministic_policy_fallback",
        )

    def _generate_with_groq(self, packet: DecisionPacket, channel: str) -> Optional[AgentInterventionResponse]:
        """Generate high-touch copy via Groq LPU API with strict timeout."""
        try:
            drivers_desc = "\n".join(
                [f"- {d.feature_name}: {d.feature_value} ({d.direction}, contribution: {d.shap_contribution:.3f})" for d in packet.top_key_drivers]
            )
            prompt = f"""
Anda adalah AI Retention Specialist untuk operator telekomunikasi (Indosat Ooredoo Hutchison).
Susun pesan retensi personalisasi dalam Bahasa Indonesia yang hangat, profesional, dan meyakinkan untuk pelanggan berikut:

DATA PELANGGAN:
- ID Pelanggan: {packet.subscriber_id}
- Rekomendasi Aksi: {packet.recommended_action}
- Estimasi Kenaikan Retensi (CATE Uplift): {packet.expected_uplift_pct:.1f}%
- Key Drivers (Penyebab Risiko & Kebutuhan):
{drivers_desc}

ATURAN:
1. Hubungkan pesan dengan key drivers secara halus tanpa istilah teknis ML.
2. Gunakan Bahasa Indonesia baku namun ramah.
3. Output HANYA JSON berformat:
{{"message": "isi pesan pelanggan", "technical_summary": "ringkasan teknis NOC"}}
"""
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a production telecom retention agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=4.0) as client:
                res = client.post(self.groq_endpoint, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    content = json.loads(data["choices"][0]["message"]["content"])
                    return AgentInterventionResponse(
                        subscriber_id=packet.subscriber_id,
                        channel=channel,
                        guardrail_passed=True,
                        guardrail_rejection_reason=None,
                        intervention_message_id_lang=content.get("message", ""),
                        technical_summary=content.get("technical_summary", ""),
                        recommended_action=packet.recommended_action,
                        intervention_cost_idr=packet.intervention_cost_idr,
                        expected_net_value_idr=packet.expected_net_value_idr,
                        source="groq_llama_70b",
                    )
        except Exception as e:
            logger.warning(f"Groq API call failed or timed out: {e}. Switching to deterministic fallback.")
        return None
