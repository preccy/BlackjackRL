from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blackjack_env import BlackjackEnv


def test_betting_phase_action_masks_and_bankroll() -> None:
    env = BlackjackEnv(
        seed=9,
        obs_version=4,
        episode_mode="shoe",
        enable_betting=True,
        bet_levels=[1.0, 2.0, 4.0],
        bankroll_start=10.0,
        bankroll_stop_on_zero=False,
        max_rounds_per_episode=20,
    )

    obs, info = env.reset(seed=19)
    assert obs.shape == (22,)
    assert env.phase == "BET"
    mask = info["action_mask"]
    assert mask.tolist() == [False, False, False, False, True, True, True]

    obs, reward, term, trunc, info = env.step(6)
    assert not term and not trunc
    assert reward == 0.0
    assert env.phase == "PLAY"
    assert env.current_bet == 4.0
    play_mask = info["action_mask"]
    assert play_mask[4:].sum() == 0
    assert play_mask[:4].sum() >= 1

    start_bankroll = env.bankroll_current
    while True:
        legal = [i for i, ok in enumerate(info["action_mask"]) if ok]
        obs, _reward, term, trunc, info = env.step(legal[0])
        assert not trunc
        if info.get("round_end"):
            break

    assert env.bankroll_current != start_bankroll
