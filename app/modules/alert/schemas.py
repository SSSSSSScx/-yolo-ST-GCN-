"""Alert DTOs."""

from pydantic import BaseModel, Field


class BatchDeleteDto(BaseModel):
    ids: list[int] = Field(min_length=1)
