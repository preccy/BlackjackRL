from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from basic_strategy_oracle import oracle_action
from blackjack_env import BlackjackEnv
from imitation_pretrain import PAIR_RANKS, parse_bet_levels
from utils.cards import Card

DEALER_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]


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


def _play_mask(can_double: bool, can_split: bool, n_actions: int) -> np.ndarray:
    mask = np.zeros(n_actions, dtype=np.float32)
    mask[:4] = [1.0, 1.0, 1.0 if can_double else 0.0, 1.0 if can_split else 0.0]
    return mask


def _load_model(path: str):
    try:
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path, device="cpu"), True
    except Exception:
        from stable_baselines3 import PPO

        return PPO.load(path, device="cpu"), False


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug PLAY policy accuracy against basic-strategy oracle.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--obs-version", type=int, default=4)
    ap.add_argument("--bet-levels", type=str, default="1,2,4,8")
    args = ap.parse_args()

    bet_levels = parse_bet_levels(args.bet_levels)
    env = BlackjackEnv(obs_version=args.obs_version, enable_betting=True, bet_levels=bet_levels, seed=7)
    model, use_mask = _load_model(args.model)

    rows: list[tuple[str, int, int]] = []

    def eval_state(label: str, cards: list[Card], dealer_rank: str) -> None:
        dealer = _card(dealer_rank, "♥")
        can_double = len(cards) == 2
        can_split = len(cards) == 2 and cards[0].rank_value == cards[1].rank_value
        obs = env.obs_from_cards(
            cards,
            dealer,
            force_can_double=can_double,
            force_can_split=can_split,
            remaining_hands=0,
            is_first_decision=True,
            phase_is_bet=False,
            cum_bins=np.zeros(10, dtype=np.float32),
            cards_remaining_norm=1.0,
            rounds_since_shuffle_norm=0.0,
        )
        mask = _play_mask(can_double, can_split, env.action_space.n)
        kwargs = dict(deterministic=True)
        if use_mask:
            kwargs["action_masks"] = mask
        pred, _ = model.predict(obs, **kwargs)
        pred = int(pred)

        tgt = oracle_action(cards, dealer_rank, can_double=can_double, can_split=can_split)
        if tgt == 2 and not can_double:
            tgt = 1
        if tgt == 3 and not can_split:
            tgt = 1
        rows.append((label, tgt, pred))

    for d in DEALER_RANKS:
        for total in range(8, 21):
            eval_state(f"hard {total} vs {d}", _hard_cards(total), d)
        for kicker in range(2, 10):
            eval_state(f"soft A{kicker} vs {d}", _soft_cards(kicker), d)
        for rank in PAIR_RANKS:
            eval_state(f"pair {rank}{rank} vs {d}", _pair_cards(rank), d)

    total = len(rows)
    correct = sum(1 for _, tgt, pred in rows if tgt == pred)
    acc = 100.0 * correct / max(1, total)
    print(f"PLAY oracle accuracy: {correct}/{total} = {acc:.2f}%")

    mismatches = [(lbl, t, p) for lbl, t, p in rows if t != p]
    if not mismatches:
        print("No mismatches.")
        return

    print("Top mismatches:")
    counts = Counter((t, p) for _, t, p in mismatches)
    for (t, p), n in counts.most_common(10):
        print(f"  oracle={t} predicted={p} count={n}")
    for lbl, t, p in mismatches[:15]:
        print(f"  {lbl}: oracle={t} predicted={p}")


if __name__ == "__main__":
    main()
