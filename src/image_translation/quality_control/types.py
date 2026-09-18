"""Shared quality-control types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class QcSeverity(str, Enum):
    error = "error"
    warning = "warning"


@dataclass
class QcIssue:
    severity: QcSeverity
    code: str
    message: str
    region_id: str = ""


@dataclass
class QcResult:
    passed: bool
    issues: List[QcIssue] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == QcSeverity.error for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == QcSeverity.warning for i in self.issues)

    def to_dict(self) -> dict:
        payload = {
            "passed": self.passed,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "code": issue.code,
                    "message": issue.message,
                    "region_id": issue.region_id,
                }
                for issue in self.issues
            ],
        }
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload
