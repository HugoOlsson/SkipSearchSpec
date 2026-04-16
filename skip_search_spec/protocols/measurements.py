from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

from skip_search_spec.helpers.versioning import get_git_revision

ExperimentType = Literal[
    "flashhead",
    "early_exit",
    "self_speculation",
    "speculative_decoding",
]
SummaryType = ExperimentType


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    experiment_type: ExperimentType
    git_commit: str
    created_at_utc: str = field(default_factory=utc_now_iso)
    git_tag: str | None = None
    model_names: tuple[str, ...] = ()
    dataset_name: str | None = None
    run_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        experiment_type: ExperimentType,
        git_commit: str | None = None,
        git_tag: str | None = None,
        created_at_utc: str | None = None,
        model_names: tuple[str, ...] = (),
        dataset_name: str | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> "RunContext":
        resolved_commit = git_commit
        resolved_tag = git_tag

        if resolved_commit is None or resolved_tag is None:
            state = get_git_revision()
            if resolved_commit is None:
                resolved_commit = state.commit
            if resolved_tag is None:
                resolved_tag = state.tag

        if resolved_commit is None:
            raise RuntimeError("Could not resolve git commit for RunContext.")

        return cls(
            run_id=run_id,
            experiment_type=experiment_type,
            git_commit=resolved_commit,
            created_at_utc=created_at_utc or utc_now_iso(),
            git_tag=resolved_tag,
            model_names=model_names,
            dataset_name=dataset_name,
            run_config=dict(run_config or {}),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunContext":
        return cls(
            run_id=str(data["run_id"]),
            experiment_type=str(data["experiment_type"]),  # type: ignore[arg-type]
            git_commit=str(data["git_commit"]),
            created_at_utc=str(data.get("created_at_utc", utc_now_iso())),
            git_tag=_opt_str(data.get("git_tag")),
            model_names=tuple(str(x) for x in list(data.get("model_names", []))),
            dataset_name=_opt_str(data.get("dataset_name")),
            run_config=dict(data.get("run_config", {})),
        )

    def print(self) -> None:
        title = self.experiment_type.replace("_", "-").upper()
        print(f"STARTING {title} RUN")
        for key, value in asdict(self).items():
            print(f"  {key}={value}")


@dataclass(frozen=True, slots=True)
class MetricEvent:
    timestamp_utc: str
    phase: str
    name: str
    value: float
    step: int | None = None
    unit: str | None = None

    @classmethod
    def create(
        cls,
        *,
        phase: str,
        name: str,
        value: float,
        step: int | None = None,
        unit: str | None = None,
    ) -> "MetricEvent":
        return cls(
            timestamp_utc=utc_now_iso(),
            phase=phase,
            name=name,
            value=float(value),
            step=step,
            unit=unit,
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
        )


@dataclass(frozen=True, slots=True)
class FlashHeadSummary:
    summary_type: Literal["flashhead"] = "flashhead"
    top1_containment: float | None = None
    top3_containment: float | None = None
    dense_winner_in_candidate_set_rate: float | None = None
    mean_candidate_count: float | None = None
    speedup_vs_baseline: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FlashHeadSummary":
        return cls(
            summary_type="flashhead",
            top1_containment=_opt_float(data.get("top1_containment")),
            top3_containment=_opt_float(data.get("top3_containment")),
            dense_winner_in_candidate_set_rate=_opt_float(data.get("dense_winner_in_candidate_set_rate")),
            mean_candidate_count=_opt_float(data.get("mean_candidate_count")),
            speedup_vs_baseline=_opt_float(data.get("speedup_vs_baseline")),
        )


@dataclass(frozen=True, slots=True)
class EarlyExitSummary:
    summary_type: Literal["early_exit"] = "early_exit"
    ce_gap_final: float | None = None
    kl_full_to_mid_mean: float | None = None
    top1_agreement_mean: float | None = None
    p_mid_on_full_argmax_mean: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EarlyExitSummary":
        return cls(
            summary_type="early_exit",
            ce_gap_final=_opt_float(data.get("ce_gap_final")),
            kl_full_to_mid_mean=_opt_float(data.get("kl_full_to_mid_mean")),
            top1_agreement_mean=_opt_float(data.get("top1_agreement_mean")),
            p_mid_on_full_argmax_mean=_opt_float(data.get("p_mid_on_full_argmax_mean")),
        )


@dataclass(frozen=True, slots=True)
class SelfSpeculationSummary:
    summary_type: Literal["self_speculation"] = "self_speculation"
    acceptance_rate_macro: float | None = None
    acceptance_rate_micro: float | None = None
    accepted_tokens_total: int | None = None
    proposed_tokens_total: int | None = None
    speedup_vs_baseline: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelfSpeculationSummary":
        return cls(
            summary_type="self_speculation",
            acceptance_rate_macro=_opt_float(data.get("acceptance_rate_macro")),
            acceptance_rate_micro=_opt_float(data.get("acceptance_rate_micro")),
            accepted_tokens_total=_opt_int(data.get("accepted_tokens_total")),
            proposed_tokens_total=_opt_int(data.get("proposed_tokens_total")),
            speedup_vs_baseline=_opt_float(data.get("speedup_vs_baseline")),
        )


@dataclass(frozen=True, slots=True)
class SpeculativeDecodingSummary:
    summary_type: Literal["speculative_decoding"] = "speculative_decoding"
    acceptance_rate_macro: float | None = None
    acceptance_rate_micro: float | None = None
    speedup_vs_baseline: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpeculativeDecodingSummary":
        return cls(
            summary_type="speculative_decoding",
            acceptance_rate_macro=_opt_float(data.get("acceptance_rate_macro")),
            acceptance_rate_micro=_opt_float(data.get("acceptance_rate_micro")),
            speedup_vs_baseline=_opt_float(data.get("speedup_vs_baseline")),
        )


RunSummary: TypeAlias = (
    FlashHeadSummary
    | EarlyExitSummary
    | SelfSpeculationSummary
    | SpeculativeDecodingSummary
)


@dataclass(frozen=True, slots=True)
class MeasurementRun:
    context: RunContext
    summary: RunSummary | None = None
    metric_events: tuple[MetricEvent, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasurementRun":
        summary_raw = data.get("summary")
        return cls(
            context=RunContext.from_dict(data["context"]),
            summary=parse_run_summary(summary_raw) if isinstance(summary_raw, Mapping) else None,
            metric_events=tuple(MetricEvent.from_dict(x) for x in list(data.get("metric_events", []))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def default_output_dir(self, *, root: str | Path = "measurements") -> Path:
        revision_key = self.context.git_tag or self.context.git_commit[:8]
        return Path(root) / revision_key / self.context.experiment_type / self.context.run_id

    def save(
        self,
        *,
        root: str | Path = "measurements",
        filename: str = "run.json",
        indent: int = 2,
    ) -> Path:
        out_dir = self.default_output_dir(root=root)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / filename
        out_path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return out_path


def parse_run_summary(data: Mapping[str, Any]) -> RunSummary:
    summary_type = str(data.get("summary_type"))
    if summary_type == "flashhead":
        return FlashHeadSummary.from_dict(data)
    if summary_type == "early_exit":
        return EarlyExitSummary.from_dict(data)
    if summary_type == "self_speculation":
        return SelfSpeculationSummary.from_dict(data)
    if summary_type == "speculative_decoding":
        return SpeculativeDecodingSummary.from_dict(data)
    raise ValueError(f"Unsupported summary_type: {summary_type}")


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
