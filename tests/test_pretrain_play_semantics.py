from __future__ import annotations

import random

from blackjack_env import BlackjackEnv
from imitation_pretrain import _append_sample_v4
from utils.cards import Card


def _first_decision_feature(obs):
    return float(obs[6])


def test_append_sample_v4_sets_is_first_decision_for_three_card_hand() -> None:
    env = BlackjackEnv(obs_version=4, enable_betting=True, bet_levels=[1.0, 2.0, 4.0, 8.0], seed=11)
    obs_out, actions_out, masks_out = [], [], []
    cards = [Card(rank="5", suit="♠"), Card(rank="4", suit="♦"), Card(rank="3", suit="♥")]

    _append_sample_v4(env, cards, "6", obs_out, actions_out, masks_out, n_actions=8, rng=random.Random(1))

    assert _first_decision_feature(obs_out[0]) == 0.0


def test_append_sample_v4_sets_is_first_decision_for_two_card_hand() -> None:
    env = BlackjackEnv(obs_version=4, enable_betting=True, bet_levels=[1.0, 2.0, 4.0, 8.0], seed=12)
    obs_out, actions_out, masks_out = [], [], []
    cards = [Card(rank="9", suit="♠"), Card(rank="2", suit="♦")]

    _append_sample_v4(env, cards, "6", obs_out, actions_out, masks_out, n_actions=8, rng=random.Random(2))

    assert _first_decision_feature(obs_out[0]) == 1.0
