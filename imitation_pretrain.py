from __future__ import annotations

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


def _append_sample(env: BlackjackEnv, cards: list[Card], dealer_rank: str, obs_out: list, actions_out: list, masks_out: list):
    dealer = _card(dealer_rank, "♥")
    can_double = len(cards) == 2
    can_split = len(cards) == 2 and cards[0].rank_value == cards[1].rank_value
    obs = env.obs_from_cards(cards, dealer, force_can_double=can_double, force_can_split=can_split)
    action = oracle_action(cards, dealer_rank, can_double=can_double, can_split=can_split)
    obs_out.append(obs)
    actions_out.append(action)
    masks_out.append(_mask_from_flags(can_double, can_split))


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


def run_basic_strategy_pretrain(
    model,
    env: BlackjackEnv,
    epochs: int = 5,
    samples: int = 200_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
) -> PretrainStats:
    if env.obs_version != 2:
        raise ValueError("Basic-strategy pretraining requires obs_version=2 (dealer Ace feature).")

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
