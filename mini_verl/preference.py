"""Preference-pair construction from scored trajectory groups.

DPO trains on pairs `(chosen, rejected)` of responses to the same prompt. This
module derives pairs from rule rewards so the existing rollout/reward stages can
feed a DPO trainer unchanged. The construction is a pure function, so a future
offline preference dataset can produce the same pairs instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """One chosen/rejected response pair sampled for the same prompt group."""

    chosen: Trajectory
    rejected: Trajectory
    group_id: str

    @property
    def reward_margin(self) -> float:
        if self.chosen.reward is None or self.rejected.reward is None:
            raise TrajectoryValidationError("reward_margin requires scored trajectories")
        return float(self.chosen.reward) - float(self.rejected.reward)


def preference_pairs(
    batch: TrajectoryBatch, *, min_reward_margin: float = 0.0
) -> tuple[PreferencePair, ...]:
    """Pick, per prompt group, the highest-reward response as chosen and the lowest as rejected.

    Groups whose reward spread is at most `min_reward_margin` produce no pair:
    a tie carries no preference signal. Ordering inside a group follows the
    batch's trajectory order, so the result is deterministic.
    """
    if min_reward_margin < 0:
        raise ValueError("min_reward_margin must be non-negative")

    pairs: list[PreferencePair] = []
    for group_id, trajectories in batch.groups().items():
        if len(trajectories) < 2:
            raise TrajectoryValidationError("preference pairing requires at least two trajectories per group")
        rewards: list[float] = []
        for trajectory in trajectories:
            if trajectory.reward is None:
                raise TrajectoryValidationError("all trajectories need a reward before preference pairing")
            rewards.append(float(trajectory.reward))
        chosen_index = max(range(len(rewards)), key=rewards.__getitem__)
        rejected_index = min(range(len(rewards)), key=rewards.__getitem__)
        if rewards[chosen_index] - rewards[rejected_index] <= min_reward_margin:
            continue
        pairs.append(
            PreferencePair(
                chosen=trajectories[chosen_index],
                rejected=trajectories[rejected_index],
                group_id=group_id,
            )
        )
    return tuple(pairs)
