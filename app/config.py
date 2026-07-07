"""Central configuration (reference: ConfigService)."""

from pydantic import BaseModel


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    records_dir: str = "records"
    db_path: str = "storage/events.db"
    camera_config_path: str = "camera_service/config.yaml"
    rules_dir: str = "danger_assessor/rules"


config = AppConfig()
