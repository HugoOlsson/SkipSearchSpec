from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True, slots=True)
class SelfSpecSpeedupEstimate:
    block_size: int
    acceptance_rate: float
    flashhead_acceptance_rate: float
    head_fraction: float
    body_fraction: float
    other_fraction: float
    body_share_of_non_head: float
    draft_body_multiplier: float
    verifier_block_cost_multiplier: float
    dense_expected_tokens_per_block: float
    flashhead_expected_tokens_per_block: float
    normal_cost_for_dense_expected_tokens: float
    normal_cost_for_flashhead_expected_tokens: float
    dense_self_spec_cost_per_block: float
    flashhead_self_spec_cost_per_block: float
    dense_speedup: float
    flashhead_speedup: float
    dense_cost_per_generated_token: float
    flashhead_cost_per_generated_token: float
    normal_cost_per_generated_token: float


def calculate_self_spec_speedup(
    *,
    block_size: int,
    acceptance_rate: float,
    head_portion: float,
    flashhead_head_speedup: float,
    flashhead_acceptance_multiplier: float,
    body_share_of_non_head: float = 0.97,
    draft_body_multiplier: float = 1.0,
    verifier_block_cost_multiplier: float = 1.0,
) -> SelfSpecSpeedupEstimate:
    """
    Estimate self-speculation speedup from coarse relative compute costs.

    Model:
      - Normal greedy decoding costs 1.0 per generated token.
      - head_portion is the fraction of that normal token cost spent in the
        language-model head.
      - body_portion is inferred as:
            (1 - head_portion) * body_share_of_non_head
      - other_portion gets the remaining non-head cost.
      - One self-spec block emits 1 verifier token plus
        block_size * acceptance_rate accepted draft tokens in expectation.
      - One self-spec block costs one verifier pass plus block_size draft passes.
      - The verifier pass is modeled as verifier_block_cost_multiplier normal
        token costs. The default is 1.0, matching the usual simplified
        speculative-decoding estimate where verification over a short block is
        treated as one parallel full-model call.
      - The draft path costs:
            head + body * draft_body_multiplier + other
        per drafted token. Use draft_body_multiplier < 1.0 to model skipped
        transformer layers in the drafter.
      - FlashHead only changes the draft head cost and multiplies the original
        acceptance rate by flashhead_acceptance_multiplier.

    head_portion and body_share_of_non_head can be fractions like 0.3 or
    percentages like 30.
    """

    head_fraction = _normalize_portion_fraction(
        head_portion,
        name="head_portion",
    )
    body_share = _normalize_portion_fraction(
        body_share_of_non_head,
        name="body_share_of_non_head",
    )
    non_head_fraction = 1.0 - head_fraction
    body_fraction = non_head_fraction * body_share
    other_fraction = non_head_fraction - body_fraction

    _validate_inputs(
        block_size=block_size,
        acceptance_rate=acceptance_rate,
        head_fraction=head_fraction,
        body_fraction=body_fraction,
        other_fraction=other_fraction,
        flashhead_head_speedup=flashhead_head_speedup,
        flashhead_acceptance_multiplier=flashhead_acceptance_multiplier,
        body_share_of_non_head=body_share,
        draft_body_multiplier=draft_body_multiplier,
        verifier_block_cost_multiplier=verifier_block_cost_multiplier,
    )

    normal_cost_per_token = 1.0
    verifier_cost_per_block = normal_cost_per_token * verifier_block_cost_multiplier
    dense_draft_cost_per_token = (
        head_fraction + body_fraction * draft_body_multiplier + other_fraction
    )
    flashhead_draft_cost_per_token = (
        head_fraction / flashhead_head_speedup
        + body_fraction * draft_body_multiplier
        + other_fraction
    )

    dense_acceptance_rate = _clamp_probability(acceptance_rate)
    flashhead_acceptance_rate = _clamp_probability(
        acceptance_rate * flashhead_acceptance_multiplier
    )

    dense_expected_tokens = 1.0 + block_size * dense_acceptance_rate
    flashhead_expected_tokens = 1.0 + block_size * flashhead_acceptance_rate

    normal_cost_for_dense_expected_tokens = dense_expected_tokens * normal_cost_per_token
    normal_cost_for_flashhead_expected_tokens = (
        flashhead_expected_tokens * normal_cost_per_token
    )

    dense_self_spec_cost = verifier_cost_per_block + block_size * dense_draft_cost_per_token
    flashhead_self_spec_cost = (
        verifier_cost_per_block + block_size * flashhead_draft_cost_per_token
    )

    return SelfSpecSpeedupEstimate(
        block_size=block_size,
        acceptance_rate=dense_acceptance_rate,
        flashhead_acceptance_rate=flashhead_acceptance_rate,
        head_fraction=head_fraction,
        body_fraction=body_fraction,
        other_fraction=other_fraction,
        body_share_of_non_head=body_share,
        draft_body_multiplier=draft_body_multiplier,
        verifier_block_cost_multiplier=verifier_block_cost_multiplier,
        dense_expected_tokens_per_block=dense_expected_tokens,
        flashhead_expected_tokens_per_block=flashhead_expected_tokens,
        normal_cost_for_dense_expected_tokens=normal_cost_for_dense_expected_tokens,
        normal_cost_for_flashhead_expected_tokens=normal_cost_for_flashhead_expected_tokens,
        dense_self_spec_cost_per_block=dense_self_spec_cost,
        flashhead_self_spec_cost_per_block=flashhead_self_spec_cost,
        dense_speedup=normal_cost_for_dense_expected_tokens / dense_self_spec_cost,
        flashhead_speedup=normal_cost_for_flashhead_expected_tokens
        / flashhead_self_spec_cost,
        dense_cost_per_generated_token=dense_self_spec_cost / dense_expected_tokens,
        flashhead_cost_per_generated_token=flashhead_self_spec_cost
        / flashhead_expected_tokens,
        normal_cost_per_generated_token=normal_cost_per_token,
    )


def _validate_inputs(
    *,
    block_size: int,
    acceptance_rate: float,
    head_fraction: float,
    body_fraction: float,
    other_fraction: float,
    flashhead_head_speedup: float,
    flashhead_acceptance_multiplier: float,
    body_share_of_non_head: float,
    draft_body_multiplier: float,
    verifier_block_cost_multiplier: float,
) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError(
            f"acceptance_rate must be between 0 and 1, got {acceptance_rate}."
        )

    portions = {
        "head_fraction": head_fraction,
        "body_fraction": body_fraction,
        "other_fraction": other_fraction,
    }
    for name, value in portions.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    if abs(sum(portions.values()) - 1.0) > 1e-9:
        raise ValueError(
            "head/body/other fractions must sum to 1.0, "
            f"got {sum(portions.values())}."
        )

    if not 0.0 <= body_share_of_non_head <= 1.0:
        raise ValueError(
            "body_share_of_non_head must be between 0 and 1, "
            f"got {body_share_of_non_head}."
        )

    if flashhead_head_speedup <= 0.0:
        raise ValueError(
            "flashhead_head_speedup must be positive, "
            f"got {flashhead_head_speedup}."
        )

    if flashhead_acceptance_multiplier < 0.0:
        raise ValueError(
            "flashhead_acceptance_multiplier must be non-negative, "
            f"got {flashhead_acceptance_multiplier}."
        )

    if draft_body_multiplier < 0.0:
        raise ValueError(
            f"draft_body_multiplier must be non-negative, got {draft_body_multiplier}."
        )

    if verifier_block_cost_multiplier <= 0.0:
        raise ValueError(
            "verifier_block_cost_multiplier must be positive, "
            f"got {verifier_block_cost_multiplier}."
        )


def _clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _normalize_portion_fraction(value: float, *, name: str) -> float:
    if 0.0 <= value <= 1.0:
        return value

    if 1.0 < value <= 100.0:
        return value / 100.0

    raise ValueError(
        f"{name} must be a fraction between 0 and 1 or a percentage between "
        f"0 and 100, got {value}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate self-speculation speedup from coarse cost portions."
    )
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--acceptance-rate",
        "--accaptence-rate",
        dest="acceptance_rate",
        type=float,
        required=True,
    )
    parser.add_argument("--head-portion", type=float, required=True)
    parser.add_argument(
        "--body-share-of-non-head",
        type=float,
        default=0.97,
        help=(
            "How much of the non-head remainder is body cost. Defaults to 0.9, "
            "so other gets 10 percent of the non-head remainder."
        ),
    )
    parser.add_argument("--flashhead-head-speedup", type=float, required=True)
    parser.add_argument(
        "--flashhead-acceptance-multiplier",
        "--flashhead-accaptence-multiplier",
        dest="flashhead_acceptance_multiplier",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--draft-body-multiplier",
        type=float,
        default=1.0,
        help="Relative body cost used by the draft path. Use <1 for skipped layers.",
    )
    parser.add_argument(
        "--verifier-block-cost-multiplier",
        type=float,
        default=1.0,
        help="Verifier block cost in normal-token cost units.",
    )

    args = parser.parse_args()
    estimate = calculate_self_spec_speedup(
        block_size=args.block_size,
        acceptance_rate=args.acceptance_rate,
        head_portion=args.head_portion,
        flashhead_head_speedup=args.flashhead_head_speedup,
        flashhead_acceptance_multiplier=args.flashhead_acceptance_multiplier,
        body_share_of_non_head=args.body_share_of_non_head,
        draft_body_multiplier=args.draft_body_multiplier,
        verifier_block_cost_multiplier=args.verifier_block_cost_multiplier,
    )
    print(json.dumps(asdict(estimate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
