import os
import yaml
from loguru import logger


class ZoneManager:
    """Per-camera danger/restricted zone definitions and point-in-zone checks.

    Each camera has its own zone config: danger_assessor/rules/zones_{camera_id}.yaml
    No zones are preset — all start empty.
    """

    def __init__(self, rules_dir: str = "danger_assessor/rules"):
        self._rules_dir = rules_dir
        self._zones: dict[str, dict[str, dict]] = {}  # camera_id -> {zone_id -> zone_def}

    def load_zones(self, camera_id: str) -> None:
        path = os.path.join(self._rules_dir, f"zones_{camera_id}.yaml")
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            zones = {}
            for zone in config.get("zones", []):
                zones[zone["id"]] = zone
            self._zones[camera_id] = zones
            logger.info(f"Loaded {len(zones)} zones for {camera_id}")
        except FileNotFoundError:
            self._zones[camera_id] = {}
            logger.info(f"No zones for {camera_id} (empty)")

    def save_zones(self, camera_id: str) -> None:
        path = os.path.join(self._rules_dir, f"zones_{camera_id}.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        zones_list = list(self._zones.get(camera_id, {}).values())
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"zones": zones_list}, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def add_zone(self, camera_id: str, zone_def: dict) -> None:
        if camera_id not in self._zones:
            self._zones[camera_id] = {}
        self._zones[camera_id][zone_def["id"]] = zone_def
        self.save_zones(camera_id)

    def delete_zone(self, camera_id: str, zone_id: str) -> bool:
        zones = self._zones.get(camera_id, {})
        if zone_id in zones:
            del zones[zone_id]
            self.save_zones(camera_id)
            return True
        return False

    def get_zones(self, camera_id: str) -> list[dict]:
        return list(self._zones.get(camera_id, {}).values())

    def get_zone_polygons(self, camera_id: str) -> list[dict]:
        """Return zone polygons for overlay drawing."""
        result = []
        for zid, zone in self._zones.get(camera_id, {}).items():
            polygon = zone.get("polygon", [])
            if polygon and len(polygon) >= 3:
                result.append({
                    "id": zid,
                    "type": zone.get("type", "restricted_zone"),
                    "polygon": polygon,
                })
        return result

    def check_position(self, camera_id: str, point: tuple[float, float]) -> list[str]:
        """Return list of zone IDs containing the point."""
        x, y = point
        result = []
        for zone_id, zone in self._zones.get(camera_id, {}).items():
            polygon = zone.get("polygon", [])
            if self._point_in_polygon(x, y, polygon):
                result.append(zone_id)
        return result

    def get_zone_type(self, camera_id: str, zone_id: str) -> str:
        zone = self._zones.get(camera_id, {}).get(zone_id, {})
        return zone.get("type", "restricted_zone")

    def get_zone_info(self, camera_id: str, zone_id: str) -> dict | None:
        return self._zones.get(camera_id, {}).get(zone_id)

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
        if not polygon or len(polygon) < 3:
            return False
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
