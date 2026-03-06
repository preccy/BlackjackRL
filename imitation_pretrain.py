from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from basic_strategy_oracle import oracle_action
from blackjack_env import BlackjackEnv
from utils.cards import Card

DEALER_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]
PAIR_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]


@dataclass
class PretrainStats:
    samples: int
    epochs: int
    final_loss: float


def _card(rank: str, suit: str = "♠") -> Card:
    return Card(rank=rank, suit=suit)


def _hard_cards(total: int) -> list[Card]:
    if total <= 11:
        return [_card("2"), _card(str(max(2, total - 2)), "♦")]
    if total == 20:
        return [_card("T"), _card("7"), _card("3")]
    if total == 19:
        return [_card("T"), _card("9")]
    return [_card("T"), _card(str(total - 10), "♦")]


def _soft_cards(kicker: int) -> list[Card]:
    return [_card("A"), _card(str(kicker), "♦")]


def _pair_cards(rank: str) -> list[Card]:
    return [_card(rank), _card(rank, "♦")]


def _mask_from_flags(can_double: bool, can_split: bool) -> np.ndarray:
    return np.array([1.0, 1.0, 1.0 if can_double else 0.0, 1.0 if can_split else 0.0], dtype=np.float32)


def parse_bet_levels(bet_levels: str | list[float]) -> list[float]:
    if isinstance(bet_levels, str):
        parsed = [float(tok.strip()) for tok in bet_levels.split(",") if tok.strip()]
    else:
        parsed = [float(v) for v in bet_levels]
    if not parsed or any(v <= 0 for v in parsed):
        raise ValueError("bet_levels must contain at least one positive value")
    return parsed


def add_pretrain_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--obs-version", type=int, choices=[2, 4], default=2)
    parser.add_argument("--enable-betting", action="store_true")
    parser.add_argument("--bet-levels", type=str, default="1,2,4,8")
    parser.add_argument("--pretrain-bet-mode", choices=["minbet"], default="minbet")
    return parser


def validate_pretrain_config(obs_version: int, enable_betting: bool, bet_levels: list[float]) -> None:
    if enable_betting:
        if obs_version != 4:
            raise ValueError("Basic-strategy pretraining with betting requires obs_version=4.")
        if len(bet_levels) < 1:
            raise ValueError("Basic-strategy pretraining with betting requires at least one bet level.")
    elif obs_version != 2:
        raise ValueError("Basic-strategy pretraining without betting requires obs_version=2.")


def _play_mask_from_flags(can_double: bool, can_split: bool, n_actions: int) -> np.ndarray:
    mask = np.zeros(n_actions, dtype=np.float32)
    mask[:4] = [1.0, 1.0, 1.0 if can_double else 0.0, 1.0 if can_split else 0.0]
    return mask


def _bet_mask(n_actions: int, n_bets: int) -> np.ndarray:
    mask = np.zeros(n_actions, dtype=np.float32)
    mask[4 : 4 + n_bets] = 1.0
    return mask


def _append_sample(env: BlackjackEnv, cards: list[Card], dealer_rank: str, obs_out: list, actions_out: list, masks_out: list):
    dealer = _card(dealer_rank, "♥")
    can_double = len(cards) == 2
    can_split = len(cards) == 2 and cards[0].rank_value == cards[1].rank_value
    obs = env.obs_from_cards(
        cards,
        dealer,
        force_can_double=can_double,
        force_can_split=can_split,
        remaining_hands=0,
        is_first_decision=(len(cards) == 2),
    )
    action = oracle_action(cards, dealer_rank, can_double=can_double, can_split=can_split)
    obs_out.append(obs)
    actions_out.append(action)
    masks_out.append(_mask_from_flags(can_double, can_split))


def _append_sample_v4(
    env: BlackjackEnv,
    cards: list[Card],
    dealer_rank: str,
    obs_out: list,
    actions_out: list,
    masks_out: list,
    n_actions: int,
    rng: random.Random,
) -> None:
    dealer = _card(dealer_rank, "♥")
    can_double = len(cards) == 2
    can_split = len(cards) == 2 and cards[0].rank_value == cards[1].rank_value
    is_first = len(cards) == 2
    remaining_hands = rng.randint(0, 3) if can_split else 0

    deck_size = float(max(1, 52 * env.n_decks))
    cards_seen = rng.randint(0, 200)
    raw_bins = np.zeros(10, dtype=np.float32)
    for _ in range(cards_seen):
        raw_bins[rng.randrange(10)] += 1.0
    cum_bins = raw_bins / deck_size
    cards_remaining_norm = max(0.05, min(1.0, 1.0 - (cards_seen / deck_size)))
    rounds_since_shuffle_norm = rng.uniform(0.0, 1.0)

    obs = env.obs_from_cards(
        cards,
        dealer,
        force_can_double=can_double,
        force_can_split=can_split,
        remaining_hands=remaining_hands,
        is_first_decision=is_first,
        phase_is_bet=False,
        cum_bins=cum_bins,
        cards_remaining_norm=cards_remaining_norm,
        rounds_since_shuffle_norm=rounds_since_shuffle_norm,
    )

    action = oracle_action(cards, dealer_rank, can_double=can_double, can_split=can_split)
    if action == 2 and not can_double:
        action = 1
    if action == 3 and not can_split:
        action = 1

    obs_out.append(obs)
    actions_out.append(action)
    masks_out.append(_play_mask_from_flags(can_double, can_split, n_actions))


def build_imitation_dataset(env: BlackjackEnv, random_samples: int, seed: int = 0):
    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    mask_list: list[np.ndarray] = []

    for dealer in DEALER_RANKS:
        for total in range(4, 21):
            _append_sample(env, _hard_cards(total), dealer, obs_list, action_list, mask_list)
        for kicker in range(2, 10):
            _append_sample(env, _soft_cards(kicker), dealer, obs_list, action_list, mask_list)
        for rank in PAIR_RANKS:
            _append_sample(env, _pair_cards(rank), dealer, obs_list, action_list, mask_list)

    rng = random.Random(seed)
    for _ in range(max(0, random_samples)):
        n_cards = 2 if rng.random() < 0.85 else 3
        cards = [env.shoe.draw() for _ in range(n_cards)]
        dealer = rng.choice(DEALER_RANKS)
        _append_sample(env, cards, dealer, obs_list, action_list, mask_list)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(action_list, dtype=np.int64)
    mask_arr = np.asarray(mask_list, dtype=np.float32)
    return obs_arr, act_arr, mask_arr


def build_imitation_dataset_with_betting(env: BlackjackEnv, total_samples: int, seed: int = 0, bet_fraction: float = 0.2):
    n_bets = len(env.bet_levels)
    n_actions = 4 + n_bets
    min_bet_action = 4

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    mask_list: list[np.ndarray] = []

    rng = random.Random(seed)

    n_bet_samples = max(1, int(max(1, total_samples) * bet_fraction))
    for idx in range(n_bet_samples):
        obs, _ = env.reset(seed=seed + idx)
        obs_list.append(obs.astype(np.float32))
        action_list.append(min_bet_action)
        mask_list.append(_bet_mask(n_actions, n_bets))

    for dealer in DEALER_RANKS:
        for total in range(4, 21):
            _append_sample_v4(env, _hard_cards(total), dealer, obs_list, action_list, mask_list, n_actions, rng)
        for kicker in range(2, 10):
            _append_sample_v4(env, _soft_cards(kicker), dealer, obs_list, action_list, mask_list, n_actions, rng)
        for rank in PAIR_RANKS:
            _append_sample_v4(env, _pair_cards(rank), dealer, obs_list, action_list, mask_list, n_actions, rng)

    while len(obs_list) < max(1, total_samples):
        n_cards = 2 if rng.random() < 0.85 else 3
        cards = [env.shoe.draw() for _ in range(n_cards)]
        dealer = rng.choice(DEALER_RANKS)
        _append_sample_v4(env, cards, dealer, obs_list, action_list, mask_list, n_actions, rng)

    obs_arr = np.asarray(obs_list[:total_samples], dtype=np.float32)
    act_arr = np.asarray(action_list[:total_samples], dtype=np.int64)
    mask_arr = np.asarray(mask_list[:total_samples], dtype=np.float32)
    return obs_arr, act_arr, mask_arr


def run_basic_strategy_pretrain(
    model,
    env: BlackjackEnv,
    epochs: int = 5,
    samples: int = 200_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
    enable_betting: bool | None = None,
    bet_levels: list[float] | None = None,
    pretrain_bet_mode: str = "minbet",
) -> PretrainStats:
    effective_enable_betting = env.enable_betting if enable_betting is None else bool(enable_betting)
    effective_bet_levels = env.bet_levels if bet_levels is None else bet_levels
    validate_pretrain_config(env.obs_version, effective_enable_betting, effective_bet_levels)
    if pretrain_bet_mode != "minbet":
        raise ValueError(f"Unsupported pretrain_bet_mode={pretrain_bet_mode}")

    if effective_enable_betting:
        obs_arr, act_arr, mask_arr = build_imitation_dataset_with_betting(env, total_samples=samples, seed=seed)
    else:
        obs_arr, act_arr, mask_arr = build_imitation_dataset(env, random_samples=samples, seed=seed)
    dataset = TensorDataset(
        torch.from_numpy(obs_arr),
        torch.from_numpy(act_arr),
        torch.from_numpy(mask_arr),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = model.device
    model.policy.train()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    final_loss = 0.0
    for _ in range(max(1, epochs)):
        loss_sum = 0.0
        count = 0
        for obs_batch, act_batch, mask_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            mask_batch = mask_batch.to(device)

            dist = model.policy.get_distribution(obs_batch)
            logits = dist.distribution.logits
            masked_logits = logits.masked_fill(mask_batch <= 0, -1e9)
            loss = F.cross_entropy(masked_logits, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            count += 1
        final_loss = loss_sum / max(1, count)

    return PretrainStats(samples=len(dataset), epochs=max(1, epochs), final_loss=final_loss)
