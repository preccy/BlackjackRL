from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blackjack_env import BlackjackEnv


def run_sanity(round_target: int = 500) -> None:
    env = BlackjackEnv(
        seed=123,
        obs_version=3,
        episode_mode="shoe",
        max_rounds_per_episode=50,
    )

    resets = 0
    total_rounds = 0
    reshuffle_terminations = 0
    maxround_terminations = 0
    saw_nonzero_bins = False
    prev_shuffle = env.shoe.shuffle_count

    obs, info = env.reset(seed=7)
    resets += 1
    assert obs.shape == (20,), f"Unexpected obs shape: {obs.shape}"

    while total_rounds < round_target:
        mask = info.get("action_mask", env.action_masks())
        legal_actions = [i for i, ok in enumerate(mask) if ok]
        action = random.choice(legal_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        assert not truncated
        assert obs.shape == (20,), f"Unexpected obs shape: {obs.shape}"

        if info.get("round_end"):
            total_rounds += 1
            if total_rounds > 1 and obs[9:19].sum() > 0:
                saw_nonzero_bins = True

        if terminated:
            if info.get("reshuffle_happened"):
                reshuffle_terminations += 1
            elif info.get("rounds_in_episode") == env.max_rounds_per_episode:
                maxround_terminations += 1
            obs, info = env.reset()
            resets += 1
            assert obs.shape == (20,), f"Unexpected obs shape after reset: {obs.shape}"

        assert env.shoe.shuffle_count >= prev_shuffle
        prev_shuffle = env.shoe.shuffle_count

    assert resets < total_rounds, "Expected shoe-mode to progress multiple rounds per reset"
    assert env.shoe.shuffle_count > 1, "Expected occasional reshuffles over long run"
    assert reshuffle_terminations + maxround_terminations > 0, "Expected termination events in shoe mode"
    assert saw_nonzero_bins, "Expected previous-round rank bins to become non-zero"

    print(
        f"OK rounds={total_rounds} resets={resets} reshuffle_terms={reshuffle_terminations} "
        f"maxround_terms={maxround_terminations} shuffle_count={env.shoe.shuffle_count}"
    )


if __name__ == "__main__":
    run_sanity()
