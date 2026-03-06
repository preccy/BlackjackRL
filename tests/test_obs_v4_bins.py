from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import random

from blackjack_env import BlackjackEnv


def test_obs_v4_cumulative_bins_and_reset_on_reshuffle() -> None:
    env = BlackjackEnv(seed=3, obs_version=4, episode_mode="shoe", max_rounds_per_episode=400)
    obs, info = env.reset(seed=11)
    assert obs.shape == (22,)

    saw_nonzero = False
    prev_shuffle = env.shoe.shuffle_count
    saw_reset_after_shuffle = False

    for _ in range(8000):
        mask = info.get("action_mask", env.action_masks())
        legal = [i for i, ok in enumerate(mask) if ok]
        action = random.choice(legal)
        obs, _reward, term, trunc, info = env.step(action)
        assert not trunc

        if info.get("round_end"):
            if float(env.cum_revealed_bins.sum()) > 0:
                saw_nonzero = True

        if env.shoe.shuffle_count > prev_shuffle:
            prev_shuffle = env.shoe.shuffle_count
            if info.get("round_end"):
                assert float(env.cum_revealed_bins.sum()) >= 0
            if float(env.cum_revealed_bins.sum()) == 0:
                saw_reset_after_shuffle = True

        if term:
            obs, info = env.reset()
            assert obs.shape == (22,)

        if saw_nonzero and saw_reset_after_shuffle:
            break

    assert saw_nonzero, "expected cumulative revealed-card bins to become non-zero"
    assert saw_reset_after_shuffle, "expected cumulative bins reset after reshuffle"
