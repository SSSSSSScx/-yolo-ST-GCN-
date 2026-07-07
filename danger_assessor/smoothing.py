import time
from collections import defaultdict
from loguru import logger


class LevelSmoother:
    """EMA smoothing + dedup window + escalation for danger levels."""

    def __init__(self, ema_alpha: float = 0.3, dedup_window: float = 10.0, escalation_time: float = 30.0):
        self._ema_alpha = ema_alpha
        self._dedup_window = dedup_window
        self._escalation_time = escalation_time
        self._ema_state: dict[tuple, float] = {}  # (track_id, rule_id) -> smoothed_level
        self._last_alert: dict[tuple, float] = {}  # (track_id, rule_id) -> last_trigger_time
        self._first_trigger: dict[tuple, float] = {}  # (track_id, rule_id) -> first_trigger_time

    def smooth(self, alerts: list[dict]) -> list[dict]:
        """Apply EMA smoothing, dedup, and escalation. L3 bypasses smoothing."""
        now = time.time()
        output = []

        # Group alerts by (track_id, rule_id)
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for alert in alerts:
            key = (alert["track_id"], alert["rule_id"])
            grouped[key].append(alert)

        for key, group in grouped.items():
            alert = group[0]
            new_level = alert["level"]
            rule_id = alert.get("rule_id", "")

            # L3 immediately bypasses smoothing
            if new_level >= 3:
                output.append(alert)
                self._ema_state[key] = new_level
                self._last_alert[key] = now
                if key not in self._first_trigger:
                    self._first_trigger[key] = now
                continue

            # EMA smoothing — first trigger uses raw level immediately
            prev = self._ema_state.get(key, 0.0)
            is_first = key not in self._last_alert

            if is_first:
                smoothed = new_level
            else:
                smoothed = self._ema_alpha * new_level + (1 - self._ema_alpha) * prev
            self._ema_state[key] = smoothed

            # Dedup
            last = self._last_alert.get(key, 0.0)
            if not is_first and now - last < self._dedup_window:
                continue

            # Escalation: sustained > escalation_time → level +1
            # Only for safety-critical rules (zone entry, running)
            ESCALATION_RULES = {"B01", "A03"}
            first = self._first_trigger.get(key, now)
            if now - first > self._escalation_time and smoothed >= 2 and rule_id in ESCALATION_RULES:
                smoothed = min(3, smoothed + 1)

            rounded_level = round(smoothed)
            if rounded_level >= 1:
                alert["level"] = rounded_level
                alert["smoothed"] = True
                output.append(alert)
                self._last_alert[key] = now
                if key not in self._first_trigger:
                    self._first_trigger[key] = now

        # Clear stale state
        stale_keys = [k for k, v in self._first_trigger.items() if now - v > 120]
        for k in stale_keys:
            self._ema_state.pop(k, None)
            self._last_alert.pop(k, None)
            self._first_trigger.pop(k, None)

        return output
