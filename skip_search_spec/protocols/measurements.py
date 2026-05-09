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
    now = time.monotonic()
    last = _LAST_SAVE_MONOTONIC_BY_RUN_ID.get(run.context.run_id)

    should_save = force or last is None or (now - last) >= min_interval_seconds
    if not should_save:
        return None

    out_path = run.save(filename=filename)
    _LAST_SAVE_MONOTONIC_BY_RUN_ID[run.context.run_id] = now
    return out_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    run_name: str
    experiment_type: ExperimentType
    git_commit: str
    created_at_utc: str = field(default_factory=utc_now_iso)
    git_tag: str | None = None
    model_names: tuple[str, ...] = ()
    dataset_name: str | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    name_comment: str | None = None

    @classmethod
    def create(
        cls,
        *,
        run_name: str,
        experiment_type: ExperimentType,
        git_commit: str | None = None,
        git_tag: str | None = None,
        created_at_utc: str | None = None,
        model_names: tuple[str, ...] = (),
        dataset_name: str | None = None,
        run_config: Mapping[str, Any] | None = None,
        name_comment: str | None = None,
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

        created_at = created_at_utc or utc_now_iso()
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        return cls(
            run_id=make_run_id(dt),
            run_name=run_name,
            experiment_type=experiment_type,
            git_commit=resolved_commit,
            created_at_utc=created_at,
            git_tag=resolved_tag,
            model_names=model_names,
            dataset_name=dataset_name,
            run_config=dict(run_config or {}),
            name_comment=name_comment,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunContext":
        return cls(
            run_id=str(data["run_id"]),
            run_name=str(data["run_name"]),
            experiment_type=str(data["experiment_type"]),  # type: ignore[arg-type]
            git_commit=str(data["git_commit"]),
            created_at_utc=str(data["created_at_utc"]),
            git_tag=_opt_str(data.get("git_tag")),
            model_names=tuple(str(x) for x in list(data.get("model_names", []))),
            dataset_name=_opt_str(data.get("dataset_name")),
            run_config=dict(data.get("run_config", {})),
            name_comment=_opt_str(data.get("name_comment")),
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
        date_key = self.context.created_at_utc[:10]
        commit_key = safe_path_part(self.context.git_commit[:6])
        date_commit_key = f"{date_key}-{commit_key}"

        run_dir_name = (
            f"{self.context.run_id}"
            f"__{safe_path_part(self.context.run_name)}"
        )
        name_comment = self.context.name_comment
        if name_comment:
            safe_comment = safe_path_part(name_comment).strip("_")
            if safe_comment:
                run_dir_name = f"{safe_comment}_{run_dir_name}"

        return (
            Path(root)
            / date_commit_key
            / self.context.experiment_type
            / run_dir_name
        )

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


def safe_path_part(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )

def print_metric_events_line(events: Iterable[MetricEvent], *, decimals: int = 4) -> None:
    for event in events:
        event.print(end="  ", decimals=decimals)
    print()

_MONTH_CODES = {
    1: "JA",
    2: "F",
    3: "MR",
    4: "AP",
    5: "MY",
    6: "JN",
    7: "JL",
    8: "AU",
    9: "S",
    10: "O",
    11: "N",
    12: "D",
}


def make_run_id(dt: datetime) -> str:
    month_code = _MONTH_CODES[dt.month]
    millisecond_prefix = f"{dt.microsecond // 1000:03d}"[:2]
    return f"{dt:%H%M%S}{millisecond_prefix}_{month_code}{dt:%d}"

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


def dataset_mix_name(dataset_mix: list[tuple[DatasetSpec, float, int]]) -> str:
    return " + ".join(
        f"{dataset_spec.name}:{weight:g}"
        for dataset_spec, weight, max_number in dataset_mix
    )


def dataset_mix_config(
    dataset_mix: list[tuple[DatasetSpec, float, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "huggingface_path": spec.huggingface_path,
            "config_name": spec.config_name,
            "split": spec.split,
            "text_field": spec.text_field,
            "weight": weight,
            "max_examples": max_examples,
        }
        for spec, weight, max_examples in dataset_mix
    ]
