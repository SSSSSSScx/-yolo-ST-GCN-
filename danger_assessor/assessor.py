from loguru import logger
from .zones import ZoneManager
from .rules_engine import RulesEngine
from .smoothing import LevelSmoother


class DangerAssessor:
    """Integrates zone management, rule engine, and level smoothing."""

    def __init__(self, zone_manager: ZoneManager = None):
        self._zone_manager = zone_manager or ZoneManager()
        self._rules_engine = RulesEngine(zone_manager=self._zone_manager)
        self._smoother = LevelSmoother(ema_alpha=0.5, dedup_window=30.0, escalation_time=30.0)

    def reload_rules(self) -> None:
        self._rules_engine.reload_configs()

    def assess(self, pipeline_output: dict, camera_id: str = "") -> list[dict]:
        """Assess danger levels for all tracked persons in a frame.

        Zones are stored in original frame coordinates. Person centroids
        are also in original frame coordinates — direct matching, no scaling.
        """
        all_alerts = []
        persons = pipeline_output.get("persons", [])
        timestamp = pipeline_output.get("timestamp", 0)
        object_detections = pipeline_output.get("object_detections", [])

        for person in persons:
            person["timestamp"] = timestamp
            person["frame"] = None
            person["camera_id"] = camera_id

            raw_alerts = self._rules_engine.check_all(person, object_detections, camera_id)
            all_alerts.extend(raw_alerts)

        # Add object-level alerts (no track_id, global scope)
        if object_detections:
            object_alerts = self._rules_engine.check_object_rules(None, object_detections)
            all_alerts.extend(object_alerts)

        # Apply smoothing and dedup
        smoothed = self._smoother.smooth(all_alerts)
        return smoothed
