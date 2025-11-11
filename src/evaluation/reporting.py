"""Aggregate evaluation artefacts for UI consumption."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.evaluation.metrics import EvaluationResult
from src.evaluation.visualizations import PlotArtifact


@dataclass
class EvaluationReport:
    model_name: str
    metrics: Dict[str, float]
    artifacts: List[PlotArtifact]

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": self.model_name,
            "metrics": self.metrics,
            "artifacts": [artifact.path.as_posix() for artifact in self.artifacts],
        }


def build_report(model_name: str, result: EvaluationResult, artifacts: List[PlotArtifact]) -> EvaluationReport:
    return EvaluationReport(model_name=model_name, metrics=result.as_dict(), artifacts=artifacts)


def consolidate_reports(reports: List[EvaluationReport]) -> Dict[str, object]:
    return {report.model_name: report.as_dict() for report in reports}
