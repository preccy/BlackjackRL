from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blackjack_env import BlackjackEnv


def test_shoe_mode_round_end_outcomes_stats() -> None:
    env = BlackjackEnv(
        seed=123,
        obs_version=3,
        episode_mode="shoe",
        max_rounds_per_episode=200,
    )

    rounds_target = 50_000
    resolved_rounds = 0
    resolved_hands_total = 0
    wins = pushes = losses = blackjacks = 0

    obs = None
    info = {}
    reset_seed = 7

    def process_round_info(round_info: dict) -> None:
        nonlocal resolved_rounds, resolved_hands_total, wins, pushes, losses, blackjacks
        outcomes = round_info.get("outcomes")
        if not round_info.get("round_end", bool(outcomes)):
            return
        outcomes = outcomes or getattr(env, "last_info", {}).get("outcomes", [])
        resolved_rounds += 1
        resolved_hands_total += len(outcomes)
        for out in outcomes:
            if out["reward"] > 0:
                wins += 1
            elif out["reward"] == 0:
                pushes += 1
            else:
                losses += 1
            if out["outcome"] in {"win_blackjack", "push_blackjack"}:
                blackjacks += 1

    while resolved_rounds < rounds_target:
        if obs is None:
            obs, info = env.reset(seed=reset_seed)
            reset_seed += 1
            process_round_info(info)

        if env.terminated or "immediate_reward" in info:
            mask = info.get("action_mask", env.action_masks())
            legal_actions = [i for i, ok in enumerate(mask) if ok]
            dummy_action = random.choice(legal_actions)
            obs, _reward, terminated, truncated, info = env.step(dummy_action)
            assert not truncated
            if terminated:
                process_round_info(info)
                obs = None
            continue

        mask = info.get("action_mask", env.action_masks())
        legal_actions = [i for i, ok in enumerate(mask) if ok]
        action = random.choice(legal_actions)

        obs, _reward, terminated, truncated, info = env.step(action)
        assert not truncated

        process_round_info(info)

        if terminated:
            obs = None

    denom = max(1, resolved_hands_total)
    win_rate = wins / denom
    push_rate = pushes / denom
    loss_rate = losses / denom
    blackjack_rate = blackjacks / denom

    assert abs((win_rate + push_rate + loss_rate) - 1.0) <= 0.05
    assert 0.03 <= blackjack_rate <= 0.06
