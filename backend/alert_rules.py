"""Shared validation for HTTP alert requests and persisted alert evaluation."""
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


class AlertRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[int] = None
    ticker: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1, max_length=64,
        pattern=r"^[A-Za-z0-9^][A-Za-z0-9.^=_-]*$",
    )]
    type: Literal["price", "volume", "earnings"] = "price"
    condition: Literal["above", "below", "crosses_above", "crosses_below", ">", "<"] = "above"
    threshold: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    # Stateless callers round-trip condition_met from the previous evaluation;
    # stored rules always get this state from the database, never the browser.
    was_triggered: bool = Field(default=False, strict=True)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value):
        return value.upper()

    @model_validator(mode="after")
    def require_threshold(self):
        if self.type in ("price", "volume") and self.threshold is None:
            raise ValueError("Price and volume alerts require a finite, nonnegative threshold")
        return self


class AlertCreate(AlertRule):
    type: Literal["price", "volume", "earnings"] = Field(default="price", alias="alert_type")
    label: Optional[str] = Field(default=None, max_length=500)
