from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Literal, Mapping


ExperimentType = Literal["early_exit", "head_approximation", "speculative_decoding"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    experiment_type: ExperimentType
    variant: str
    git_commit: str
    created_at_utc: str = field(default_factory=utc_now_iso)
    git_tag: str | None = None
    model_names: tuple[str, ...] = ()
    dataset_name: str | None = None
    hardware_name: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunContext":
        return cls(
            run_id=str(data["run_id"]),
            experiment_type=str(data["experiment_type"]),  # type: ignore[arg-type]
            variant=str(data["variant"]),
            git_commit=str(data["git_commit"]),
            created_at_utc=str(data.get("created_at_utc", utc_now_iso())),
            git_tag=_opt_str(data.get("git_tag")),
            model_names=tuple(str(x) for x in list(data.get("model_names", []))),
            dataset_name=_opt_str(data.get("dataset_name")),
            hardware_name=_opt_str(data.get("hardware_name")),
            notes=_opt_str(data.get("notes")),
        )


@dataclass(frozen=True, slots=True)
class MetricEvent:
    timestamp_utc: str
    phase: str
    name: str
    value: float
    step: int | None = None
    unit: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        phase: str,
        name: str,
        value: float,
        step: int | None = None,
        unit: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> "MetricEvent":
        return cls(
            timestamp_utc=utc_now_iso(),
            phase=phase,
            name=name,
            value=float(value),
            step=step,
            unit=unit,
            tags=dict(tags or {}),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricEvent":
        return cls(
            timestamp_utc=str(data["timestamp_utc"]),
            phase=str(data["phase"]),
            name=str(data["name"]),
            value=float(data["value"]),
            step=_opt_int(data.get("step")),
            unit=_opt_str(data.get("unit")),
            tags={str(k): str(v) for k, v in dict(data.get("tags", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseResult":
        return cls(
            case_id=str(data["case_id"]),
            params=dict(data.get("params", {})),
            metrics={str(k): float(v) for k, v in dict(data.get("metrics", {})).items()},
            notes=_opt_str(data.get("notes")),
        )


@dataclass(frozen=True, slots=True)
class RunSummary:
    # Keep this compact and cross-experiment. Use optional fields.
    ce_gap_final: float | None = None
    kl_full_to_mid_mean: float | None = None
    acceptance_rate_macro: float | None = None
    acceptance_rate_micro: float | None = None
    top1_containment: float | None = None
    speedup_vs_baseline: float | None = None
    tokens_per_second_baseline: float | None = None
    tokens_per_second_variant: float | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunSummary":
        return cls(
            ce_gap_final=_opt_float(data.get("ce_gap_final")),
            kl_full_to_mid_mean=_opt_float(data.get("kl_full_to_mid_mean")),
            acceptance_rate_macro=_opt_float(data.get("acceptance_rate_macro")),
            acceptance_rate_micro=_opt_float(data.get("acceptance_rate_micro")),
            top1_containment=_opt_float(data.get("top1_containment")),
            speedup_vs_baseline=_opt_float(data.get("speedup_vs_baseline")),
            tokens_per_second_baseline=_opt_float(data.get("tokens_per_second_baseline")),
            tokens_per_second_variant=_opt_float(data.get("tokens_per_second_variant")),
            notes=_opt_str(data.get("notes")),
        )


@dataclass(frozen=True, slots=True)
class MeasurementRun:
    context: RunContext
    summary: RunSummary | None = None
    metric_events: tuple[MetricEvent, ...] = ()
    case_results: tuple[CaseResult, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasurementRun":
        return cls(
            context=RunContext.from_dict(data["context"]),
            summary=RunSummary.from_dict(data["summary"]) if "summary" in data and data["summary"] is not None else None,
            metric_events=tuple(MetricEvent.from_dict(x) for x in list(data.get("metric_events", []))),
            case_results=tuple(CaseResult.from_dict(x) for x in list(data.get("case_results", []))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

