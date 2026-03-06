from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blackjack_env import BlackjackEnv
from utils.cards import Card


def _rig_next_round_dealer_blackjack(env: BlackjackEnv, player_cards: tuple[str, str]) -> None:
    """Arrange next four draws as P1, D1, P2, D2 from Shoe.pop() order."""
    p1, p2 = player_cards
    forced_draws = [
        Card(rank=p1, suit="♠"),
        Card(rank="A", suit="♥"),
        Card(rank=p2, suit="♦"),
        Card(rank="T", suit="♣"),
    ]
    env.shoe.cards.extend(reversed(forced_draws))


def test_bet_step_immediately_resolves_dealer_peek_blackjack_loss() -> None:
    env = BlackjackEnv(
        seed=11,
        obs_version=4,
        episode_mode="shoe",
        enable_betting=True,
        bet_levels=[1.0, 2.0, 4.0, 8.0],
        max_rounds_per_episode=5,
        shuffle_on_reset=False,
    )

    obs, info = env.reset(seed=101)
    assert info["phase"] == "BET"
    _rig_next_round_dealer_blackjack(env, ("9", "7"))

    obs, reward, term, trunc, info = env.step(4)
    assert not term and not trunc
    assert info.get("round_end") is True
    assert info["phase"] == "BET"
    assert info["action_mask"].tolist() == [False, False, False, False, True, True, True, True]
    assert any(outcome["outcome"] == "loss_dealer_blackjack" for outcome in info["outcomes"])
    assert info["total_wagered"] == 1.0
    assert info["current_bet"] == 1.0
    assert info["actions_taken"] == [[]]
    assert info["player_hands"][0] == [{"rank": "9", "suit": "♠", "value": 9}, {"rank": "7", "suit": "♦", "value": 7}]
    assert reward == -1.0


def test_bet_step_immediately_resolves_push_blackjack_vs_dealer_blackjack() -> None:
    env = BlackjackEnv(
        seed=12,
        obs_version=4,
        episode_mode="shoe",
        enable_betting=True,
        bet_levels=[1.0, 2.0, 4.0, 8.0],
        max_rounds_per_episode=5,
        shuffle_on_reset=False,
    )

    _, info = env.reset(seed=202)
    assert info["phase"] == "BET"
    _rig_next_round_dealer_blackjack(env, ("A", "T"))

    _, reward, term, trunc, info = env.step(4)
    assert not term and not trunc
    assert info.get("round_end") is True
    assert info["phase"] == "BET"
    assert any(outcome["outcome"] == "push_blackjack" for outcome in info["outcomes"])
    assert info["total_wagered"] == 1.0
    assert info["current_bet"] == 1.0
    assert info["actions_taken"] == [[]]
    assert reward == 0.0
