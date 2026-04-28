from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable, Literal, Mapping, TypeAlias

from skip_search_spec.helpers.versioning import get_git_revision
from skip_search_spec.protocols.windows import DatasetSpec

ExperimentType = Literal[
    "flashhead",
    "early_exit",
    "self_speculation",
    "speculative_decoding",
    "middle_gap_skip"
]
SummaryType = ExperimentType


_LAST_SAVE_MONOTONIC_BY_RUN_ID: dict[str, float] = {}


def save_at_interval(
    run: MeasurementRun,
    *,
    min_interval_seconds: float = 60.0,
    filename: str = "run.json",
    force: bool = False,
) -> Path | None:
    run_id = run.context.run_id
    now = time.monotonic()
    last = _LAST_SAVE_MONOTONIC_BY_RUN_ID.get(run_id)

    should_save = force or last is None or (now - last) >= min_interval_seconds
    if not should_save:
        return None

    out_path = run.save(filename=filename)
    _LAST_SAVE_MONOTONIC_BY_RUN_ID[run_id] = now
    return out_path


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
            if isinstance(value, Mapping):
                print(f"  {key}=")
                pretty = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
                for line in pretty.splitlines():
                    print(f"    {line}")
            else:
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

    def format_compact(self, *, decimals: int = 4) -> str:
        return f"{self.name}={self.value:.{decimals}f}"

    def print(self, *, end: str = "  ", decimals: int = 4) -> None:
        print(self.format_compact(decimals=decimals), end=end)



@dataclass(slots=True)
class MeasurementRun:
    context: RunContext
    metric_events: list[MetricEvent] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasurementRun":
        return cls(
            context=RunContext.from_dict(data["context"]),
            metric_events=[MetricEvent.from_dict(x) for x in list(data.get("metric_events", []))],
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



def print_metric_events_line(events: Iterable[MetricEvent], *, decimals: int = 4) -> None:
    for event in events:
        event.print(end="  ", decimals=decimals)
    print()


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



def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, tuple | list):
        return [json_safe(v) for v in value]

    return str(value)


def dataset_mix_name(dataset_mix: list[tuple[DatasetSpec, float]]) -> str:
    return " + ".join(
        f"{dataset_spec.name}:{weight:g}"
        for dataset_spec, weight in dataset_mix
    )


def dataset_mix_config(
    dataset_mix: list[tuple[DatasetSpec, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": dataset_spec.name,
            "huggingface_path": dataset_spec.huggingface_path,
            "config_name": dataset_spec.config_name,
            "split": dataset_spec.split,
            "text_field": dataset_spec.text_field,
            "weight": float(weight),
        }
        for dataset_spec, weight in dataset_mix
    ]