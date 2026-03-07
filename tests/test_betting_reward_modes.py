import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math

from blackjack_env import BlackjackEnv


def test_betting_reward_modes_calculation_helpers():
    env = BlackjackEnv(enable_betting=True, bet_levels=[1, 2, 4], bankroll_start=200, betting_reward_mode="roi")
    assert env._round_reward_from_mode(4.0, 8.0, 200.0, 204.0) == 0.5

    env.betting_reward_mode = "net"
    assert env._round_reward_from_mode(4.0, 8.0, 200.0, 204.0) == 4.0

    env.betting_reward_mode = "log_bankroll"
    got = env._round_reward_from_mode(4.0, 8.0, 200.0, 204.0)
    assert math.isclose(got, math.log(204.0 + 1e-8) - math.log(200.0 + 1e-8), rel_tol=1e-9)


def test_bet_entropy_bonus_scaled_by_bet_index():
    env = BlackjackEnv(enable_betting=True, bet_levels=[1, 2, 4, 8], bet_entropy_bonus=0.2)
    env.current_bet_index = 0
    assert env._bet_exploration_bonus() == 0.0
    env.current_bet_index = 3
    assert env._bet_exploration_bonus() == 0.2


def test_round_payload_exposes_wager_and_bankroll_fields():
    env = BlackjackEnv(enable_betting=True, bet_levels=[1, 2], bankroll_start=50, betting_reward_mode="roi")
    env._bankroll_before_round = 50.0
    env._bankroll_after_round = 52.0
    env.current_bet_index = 1
    env.last_info = {
        "outcomes": [],
        "total_reward": 2.0,
        "total_wagered": 4.0,
        "shoe_meta": {},
    }
    payload = env._build_round_end_payload(round_reward=0.5, reshuffle_happened=False)
    assert payload["total_wagered_this_round"] == 4.0
    assert payload["bankroll_before_round"] == 50.0
    assert payload["bankroll_after_round"] == 52.0
    assert payload["current_bet_index"] == 1


def test_non_betting_mode_uses_net_reward_behavior():
    env = BlackjackEnv(enable_betting=False)
    assert env._round_reward_from_mode(3.0, 10.0, None, None) == 3.0
