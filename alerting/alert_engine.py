"""
Alerting Engine – threshold-based alert triggering, webhook notifications,
and alert storage.

This module is designed to run as a periodic task (e.g., via APScheduler or
a Fabric notebook scheduled trigger) that evaluates incoming sensor readings
against configurable thresholds and ML model outputs.

Reference:
  https://learn.microsoft.com/en-us/azure/azure-monitor/alerts/alerts-overview
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger("alerting.engine")


# ---------------------------------------------------------------------------
# Enums & value objects
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleType(str, Enum):
    THRESHOLD_HIGH = "threshold_high"
    THRESHOLD_LOW = "threshold_low"
    ANOMALY_SCORE = "anomaly_score"
    FAILURE_PROBABILITY = "failure_probability"
    RATE_OF_CHANGE = "rate_of_change"


@dataclass
class AlertRule:
    """A single alerting rule definition."""
    rule_id: str
    sensor_name: str
    rule_type: RuleType
    warning_threshold: float
    critical_threshold: float
    description: str = ""
    cooldown_minutes: int = 15
    enabled: bool = True


@dataclass
class AlertEvent:
    """Immutable record of a triggered alert."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    turbine_id: str = ""
    sensor_name: str = ""
    severity: Severity = Severity.INFO
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# Default rule set for SGT-400 Gas Turbine
# Thresholds derived from Siemens technical documentation & OEM guidance
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[AlertRule] = [
    AlertRule(
        rule_id="EGT_HIGH",
        sensor_name="exhaust_gas_temp_c",
        rule_type=RuleType.THRESHOLD_HIGH,
        warning_threshold=540.0,
        critical_threshold=560.0,
        description="Exhaust Gas Temperature exceeds safe operating range",
    ),
    AlertRule(
        rule_id="VIB_HIGH",
        sensor_name="vibration_mm_s",
        rule_type=RuleType.THRESHOLD_HIGH,
        warning_threshold=10.0,
        critical_threshold=15.0,
        description="Vibration level above acceptable limit",
    ),
    AlertRule(
        rule_id="BEARING_TEMP",
        sensor_name="bearing_temp_c",
        rule_type=RuleType.THRESHOLD_HIGH,
        warning_threshold=110.0,
        critical_threshold=120.0,
        description="Bearing temperature elevated – possible lubrication issue",
    ),
    AlertRule(
        rule_id="LUBE_PRESS_LOW",
        sensor_name="lube_oil_pressure_bar",
        rule_type=RuleType.THRESHOLD_LOW,
        warning_threshold=2.0,
        critical_threshold=1.5,
        description="Lube oil pressure dropping below minimum",
    ),
    AlertRule(
        rule_id="POWER_DEGRADE",
        sensor_name="power_output_mw",
        rule_type=RuleType.THRESHOLD_LOW,
        warning_threshold=11.0,
        critical_threshold=9.0,
        description="Power output below expected baseline – possible degradation",
    ),
    AlertRule(
        rule_id="ANOMALY",
        sensor_name="anomaly_score",
        rule_type=RuleType.ANOMALY_SCORE,
        warning_threshold=0.5,
        critical_threshold=0.7,
        description="ML anomaly model detected unusual pattern",
    ),
    AlertRule(
        rule_id="FAILURE_PROB",
        sensor_name="failure_probability",
        rule_type=RuleType.FAILURE_PROBABILITY,
        warning_threshold=0.3,
        critical_threshold=0.6,
        description="Predicted failure probability exceeds safe limit",
    ),
]


# ---------------------------------------------------------------------------
# Webhook dispatcher
# ---------------------------------------------------------------------------

class WebhookNotifier:
    """Sends alert payloads to configured webhook endpoints (Teams, Slack, etc.)."""

    def __init__(self, webhook_urls: list[str] | None = None, timeout: float = 10.0):
        self.webhook_urls = webhook_urls or []
        self.timeout = timeout

    async def send(self, event: AlertEvent) -> list[bool]:
        """Post the alert event to every registered webhook. Returns success flags."""
        if not self.webhook_urls:
            return []

        payload = self._build_payload(event)
        results: list[bool] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for url in self.webhook_urls:
                try:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    results.append(True)
                    logger.info("Webhook sent to %s for alert %s", url, event.alert_id)
                except Exception:
                    logger.exception("Webhook failed for %s", url)
                    results.append(False)

        return results

    @staticmethod
    def _build_payload(event: AlertEvent) -> dict[str, Any]:
        """Build an Adaptive Card payload for Microsoft Teams."""
        color = {
            Severity.INFO: "0078D4",
            Severity.WARNING: "FFC107",
            Severity.HIGH: "FF8C00",
            Severity.CRITICAL: "D13438",
        }.get(event.severity, "888888")

        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "size": "Medium",
                                "weight": "Bolder",
                                "text": f"⚠️ {event.severity.value} Alert – {event.sensor_name}",
                                "color": "Attention" if event.severity in (Severity.HIGH, Severity.CRITICAL) else "Default",
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Turbine", "value": event.turbine_id},
                                    {"title": "Sensor", "value": event.sensor_name},
                                    {"title": "Value", "value": f"{event.value:.3f}"},
                                    {"title": "Threshold", "value": f"{event.threshold:.3f}"},
                                    {"title": "Time", "value": event.timestamp},
                                ],
                            },
                            {
                                "type": "TextBlock",
                                "text": event.message,
                                "wrap": True,
                            },
                        ],
                        "msteams": {"width": "Full"},
                    },
                }
            ],
        }


# ---------------------------------------------------------------------------
# Alert Engine
# ---------------------------------------------------------------------------

class AlertEngine:
    """
    Evaluates sensor data against alert rules and emits AlertEvents.

    Supports:
      - Static threshold rules (high / low)
      - ML-derived score rules (anomaly, failure probability)
      - Cooldown to avoid alert flooding
      - Webhook notification dispatch
    """

    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        notifier: WebhookNotifier | None = None,
    ):
        self.rules = {r.rule_id: r for r in (rules or DEFAULT_RULES)}
        self.notifier = notifier or WebhookNotifier()
        self._last_fired: dict[str, datetime] = {}  # rule_id -> last fire time
        self._history: list[AlertEvent] = []

    # ---- public API ----

    async def evaluate(
        self,
        turbine_id: str,
        sensor_data: dict[str, float],
        ml_scores: dict[str, float] | None = None,
    ) -> list[AlertEvent]:
        """
        Evaluate all enabled rules against the latest sensor data.

        Parameters
        ----------
        turbine_id : str
        sensor_data : dict mapping sensor_name → current value
        ml_scores : dict with keys like 'anomaly_score', 'failure_probability'

        Returns
        -------
        list of newly triggered AlertEvent objects.
        """
        now = datetime.now(timezone.utc)
        merged = {**sensor_data, **(ml_scores or {})}
        events: list[AlertEvent] = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            value = merged.get(rule.sensor_name)
            if value is None:
                continue

            severity = self._check_rule(rule, value)
            if severity is None:
                continue

            # Cooldown check
            last = self._last_fired.get(rule.rule_id)
            if last and (now - last).total_seconds() < rule.cooldown_minutes * 60:
                continue

            threshold = (
                rule.critical_threshold
                if severity == Severity.CRITICAL
                else rule.warning_threshold
            )

            event = AlertEvent(
                rule_id=rule.rule_id,
                turbine_id=turbine_id,
                sensor_name=rule.sensor_name,
                severity=severity,
                value=value,
                threshold=threshold,
                message=rule.description,
            )

            events.append(event)
            self._last_fired[rule.rule_id] = now
            self._history.append(event)

            logger.warning(
                "Alert triggered: %s | %s | %s = %.3f (threshold %.3f)",
                event.severity.value,
                rule.rule_id,
                rule.sensor_name,
                value,
                threshold,
            )

        # Fire webhooks asynchronously for each event
        for evt in events:
            await self.notifier.send(evt)

        return events

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent alert events as dicts."""
        return [asdict(e) for e in self._history[-limit:]]

    def acknowledge(self, alert_id: str, acknowledged_by: str = "operator") -> bool:
        """Mark an alert as acknowledged."""
        for event in reversed(self._history):
            if event.alert_id == alert_id:
                event.acknowledged = True
                logger.info("Alert %s acknowledged by %s", alert_id, acknowledged_by)
                return True
        return False

    def add_rule(self, rule: AlertRule) -> None:
        self.rules[rule.rule_id] = rule

    def disable_rule(self, rule_id: str) -> None:
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    def enable_rule(self, rule_id: str) -> None:
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    # ---- internal ----

    @staticmethod
    def _check_rule(rule: AlertRule, value: float) -> Severity | None:
        """Return severity if value violates the rule, else None."""
        if rule.rule_type in (
            RuleType.THRESHOLD_HIGH,
            RuleType.ANOMALY_SCORE,
            RuleType.FAILURE_PROBABILITY,
        ):
            if value >= rule.critical_threshold:
                return Severity.CRITICAL
            if value >= rule.warning_threshold:
                return Severity.WARNING
            return None

        if rule.rule_type == RuleType.THRESHOLD_LOW:
            if value <= rule.critical_threshold:
                return Severity.CRITICAL
            if value <= rule.warning_threshold:
                return Severity.WARNING
            return None

        if rule.rule_type == RuleType.RATE_OF_CHANGE:
            # Rate of change uses absolute value comparison
            if abs(value) >= rule.critical_threshold:
                return Severity.CRITICAL
            if abs(value) >= rule.warning_threshold:
                return Severity.HIGH
            return None

        return None


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_alert_engine(
    webhook_urls: list[str] | None = None,
    custom_rules: list[dict[str, Any]] | None = None,
) -> AlertEngine:
    """
    Convenience factory that builds an AlertEngine with optional custom rules.

    Parameters
    ----------
    webhook_urls : list of webhook URLs (e.g., Teams incoming webhook)
    custom_rules : list of dicts that can be unpacked into AlertRule
    """
    notifier = WebhookNotifier(webhook_urls=webhook_urls)

    rules = list(DEFAULT_RULES)
    if custom_rules:
        for rd in custom_rules:
            rd["rule_type"] = RuleType(rd["rule_type"])
            rules.append(AlertRule(**rd))

    return AlertEngine(rules=rules, notifier=notifier)
