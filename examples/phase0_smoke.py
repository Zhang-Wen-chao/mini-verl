"""Run the data and reward stages of one tiny GRPO iteration without model dependencies."""

from mini_verl.protocol import Trajectory, TrajectoryBatch
from mini_verl.reward import apply_rewards, exact_answer_reward, group_relative_advantages


def main() -> None:
    prompt = "What is 20 + 22?"
    samples = (
        Trajectory(
            prompt_token_ids=(1, 2, 3),
            response_token_ids=(4, 5),
            old_logprobs=(-0.2, -0.1),
            policy_version=0,
            group_id="addition-42",
            prompt_text=prompt,
            response_text="Reasoning omitted.\nFinal answer: 42",
            metadata={"expected_answer": "42"},
        ),
        Trajectory(
            prompt_token_ids=(1, 2, 3),
            response_token_ids=(6,),
            old_logprobs=(-0.6,),
            policy_version=0,
            group_id="addition-42",
            prompt_text=prompt,
            response_text="Final answer: 41",
            metadata={"expected_answer": "42"},
        ),
    )
    scored = apply_rewards(TrajectoryBatch(samples), exact_answer_reward)
    batch = group_relative_advantages(scored)

    for trajectory in batch.trajectories:
        print(
            f"response={final_line(trajectory.response_text)!r} reward={trajectory.reward:.1f} "
            f"advantage={trajectory.advantage:+.4f}"
        )
    print(f"trainable_response_tokens={batch.response_token_count}")


def final_line(text: str | None) -> str:
    return (text or "").splitlines()[-1]


if __name__ == "__main__":
    main()
