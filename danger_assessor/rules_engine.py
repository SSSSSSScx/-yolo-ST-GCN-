import time
from loguru import logger
from .zones import ZoneManager


class RulesEngine:
    """Three-category rule engine: Action (A), Behavior (B), Object (O)."""

    def __init__(self, zone_manager: ZoneManager = None,
                 action_config: str = "danger_assessor/rules/action_rules.yaml",
                 object_config: str = "danger_assessor/rules/object_rules.yaml"):
        self._zone_manager = zone_manager or ZoneManager()
        self._action_thresholds = self._load_yaml(action_config) or {}
        self._object_thresholds = self._load_yaml(object_config) or {}

    def reload_configs(self) -> None:
        self._action_thresholds = self._load_yaml("danger_assessor/rules/action_rules.yaml") or {}
        self._object_thresholds = self._load_yaml("danger_assessor/rules/object_rules.yaml") or {}
        logger.info("Rule configs reloaded")

    @staticmethod
    def _load_yaml(path: str) -> dict | None:
        import yaml
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Rule config not found: {path}")
            return None

    def check_all(self, person_data: dict, object_detections: list = None,
                  camera_id: str = "") -> list[dict]:
        """Run all per-person rules (A + B). Object rules are global and called separately."""
        alerts = []

        alerts.extend(self.check_action_rules(person_data))
        alerts.extend(self.check_behavior_rules(person_data, camera_id))

        return alerts

    def check_action_rules(self, person_data: dict) -> list[dict]:
        """A-category: action-based rules."""
        alerts = []
        action = person_data.get("action", "")
        track_id = person_data.get("track_id", -1)
        ts = person_data.get("timestamp", time.time())

        # A01: Eating in lab area
        if action == "饮食动作":
            cfg = self._action_thresholds.get("A01", {})
            if cfg.get("enabled", True):
                alerts.append({
                    "rule_id": "A01", "track_id": track_id, "level": 2,
                    "message": f"检测到饮食动作 (Track {track_id})", "timestamp": ts,
                })

        # A02: Pushing/roughhousing
        if action == "推搡嬉闹":
            cfg = self._action_thresholds.get("A02", {})
            if cfg.get("enabled", True):
                alerts.append({
                    "rule_id": "A02", "track_id": track_id, "level": 1,
                    "message": f"检测到推搡嬉闹行为 (Track {track_id})", "timestamp": ts,
                })

        # A03: Running
        if action == "奔跑":
            cfg = self._action_thresholds.get("A03", {})
            if cfg.get("enabled", True):
                alerts.append({
                    "rule_id": "A03", "track_id": track_id, "level": 1,
                    "message": f"检测到奔跑行为 (Track {track_id})", "timestamp": ts,
                })

        # A04: Smoking
        if action == "抽烟":
            cfg = self._action_thresholds.get("A04", {})
            if cfg.get("enabled", True):
                alerts.append({
                    "rule_id": "A04", "track_id": track_id, "level": 2,
                    "message": f"检测到抽烟行为 (Track {track_id})", "timestamp": ts,
                })

        # A05: Falling / Fallen
        if action in ("摔倒", "倒地不动"):
            cfg = self._action_thresholds.get("A05", {})
            if cfg.get("enabled", True):
                level = 3  # highest priority
                alerts.append({
                    "rule_id": "A05", "track_id": track_id, "level": level,
                    "message": f"检测到人员摔倒 (Track {track_id})", "timestamp": ts,
                })

        return alerts

    def check_behavior_rules(self, person_data: dict, camera_id: str = "") -> list[dict]:
        """B-category: spatial/behavior rules."""
        alerts = []
        track_id = person_data.get("track_id", -1)
        centroid = person_data.get("centroid", [0, 0])
        bbox = person_data.get("bbox", [])
        ts = person_data.get("timestamp", time.time())

        # B01: Entering danger zone (per-camera)
        # Check if ANY point of the person overlaps the zone:
        #   centroid + all 4 bbox corners
        test_points = [tuple(centroid)]
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            test_points += [(x1, y1), (x2, y1), (x1, y2), (x2, y2), ((x1+x2)/2, (y1+y2)/2)]

        hit_zones: dict[str, str] = {}
        for pt in test_points:
            for zone_id in self._zone_manager.check_position(camera_id, pt):
                if zone_id not in hit_zones:
                    hit_zones[zone_id] = self._zone_manager.get_zone_type(camera_id, zone_id)

        for zone_id, zone_type in hit_zones.items():
            if zone_type == "danger_zone":
                alerts.append({
                    "rule_id": "B01", "track_id": track_id, "level": 2,
                    "message": f"人员闯入高危禁区 {zone_id} (Track {track_id})", "timestamp": ts,
                })
            elif zone_type == "restricted_zone":
                alerts.append({
                    "rule_id": "B01", "track_id": track_id, "level": 2,
                    "message": f"人员进入限制区域 {zone_id} (Track {track_id})", "timestamp": ts,
                })

        # B02: Missing PPE (only for confirmed tracks)
        ppe = person_data.get("ppe", {})
        if ppe:
            missing = [k for k, v in ppe.items() if not v]
            if missing:
                alerts.append({
                    "rule_id": "B02", "track_id": track_id, "level": 1,
                    "message": f"防护装备缺失: {', '.join(missing)} (Track {track_id})", "timestamp": ts,
                })

        return alerts

    def check_object_rules(self, frame, object_detections: list) -> list[dict]:
        """O-category: object detection rules."""
        alerts = []
        ts = time.time()
        for obj in object_detections:
            cls = obj.get("class", "")
            conf = obj.get("confidence", 0)

            if cls == "smoke":
                cfg = self._object_thresholds.get("O01", {})
                if cfg.get("enabled", True) and conf >= cfg.get("threshold", 0.5):
                    alerts.append({
                        "rule_id": "O01", "track_id": -1, "level": 3,
                        "message": "检测到烟雾", "timestamp": ts,
                    })
            elif cls == "fire":
                cfg = self._object_thresholds.get("O02", {})
                if cfg.get("enabled", True):
                    alerts.append({
                        "rule_id": "O02", "track_id": -1, "level": 3,
                        "message": "检测到火焰!", "timestamp": ts,
                    })
            elif cls == "person_down":
                cfg = self._object_thresholds.get("O03", {})
                if cfg.get("enabled", True):
                    alerts.append({
                        "rule_id": "O03", "track_id": -1, "level": 3,
                        "message": "检测到人员倒地", "timestamp": ts,
                    })

        return alerts
