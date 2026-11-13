from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Evidence(BaseModel):
    tool: str
    call_id: str
    content: str


class SpecialistResult(BaseModel):
    status: Literal["completed", "needs_input", "partial", "failed"]
    summary: str = Field(min_length=1)
    findings: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_action: Literal["present_result", "request_clarification", "resume_investigation"]

    @model_validator(mode="after")
    def consistent_status(self):
        if self.status == "needs_input" and self.next_action != "request_clarification":
            raise ValueError("needs_input requires request_clarification")
        if self.status == "failed" and not self.limitations:
            raise ValueError("failed requires a limitation")
        return self