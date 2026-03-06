from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from basic_strategy_oracle import oracle_action
from blackjack_env import BlackjackEnv
from utils.cards import Card


@dataclass
class CanonicalState:
    label: str
    player_cards: list[Card]
    dealer_upcard: Card


def _meta_paths_for_model(model_path: str) -> list[Path]:
    p = Path(model_path)
    candidates = []
    if p.suffix:
        candidates.append(p.with_suffix(p.suffix + ".meta.json"))
        candidates.append(p.with_suffix(".meta.json"))
    else:
        candidates.append(Path(f"{model_path}.meta.json"))
    return candidates


def _resolve_obs_version(model_path: str, obs_version_arg: int | None) -> int:
    if obs_version_arg is not None:
        return obs_version_arg
    for meta_path in _meta_paths_for_model(model_path):
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return int(meta.get("env", {}).get("obs_version", 4))
        except Exception:
            break
    return 4


def _build_states() -> list[CanonicalState]:
    states: list[CanonicalState] = []
    dealer_ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]

    for dealer in dealer_ranks:
        dealer_card = Card(rank=dealer, suit="♠")

        for total in range(8, 21):
            c1 = max(2, total - 10)
            c2 = total - c1
            if c2 > 10:
                c1 = total - 10
                c2 = 10
            cards = [Card(rank=str(c1), suit="♥"), Card(rank="T" if c2 == 10 else str(c2), suit="♦")]
            states.append(CanonicalState(label=f"hard {total}", player_cards=cards, dealer_upcard=dealer_card))

        for kicker in range(2, 10):
            cards = [Card(rank="A", suit="♥"), Card(rank=str(kicker), suit="♦")]
            states.append(CanonicalState(label=f"soft A{kicker}", player_cards=cards, dealer_upcard=dealer_card))

        for pair in ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]:
            cards = [Card(rank=pair, suit="♥"), Card(rank=pair, suit="♦")]
            pair_label = "TT" if pair == "T" else ("AA" if pair == "A" else f"{pair}{pair}")
            states.append(CanonicalState(label=f"pair {pair_label}", player_cards=cards, dealer_upcard=dealer_card))

    return states


def _build_play_mask(env: BlackjackEnv, cards: list[Card]) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[:4] = True
    can_double = len(cards) == 2
    can_split = len(cards) == 2 and cards[0].rank_value == cards[1].rank_value
    mask[2] = can_double
    mask[3] = can_split
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--algo", choices=["auto", "maskable", "ppo"], default="auto")
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--show-mismatches", type=int, default=20)
    args = parser.parse_args()

    model = None
    using_maskable = False
    if args.algo in {"auto", "maskable"}:
        try:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(args.model)
            using_maskable = True
        except Exception:
            if args.algo == "maskable":
                raise

    if model is None:
        model = PPO.load(args.model)

    obs_version = _resolve_obs_version(args.model, args.obs_version)
    env = BlackjackEnv(obs_version=obs_version, enable_betting=True, bet_levels=[1.0, 2.0, 4.0, 8.0])

    states = _build_states()
    mismatches: list[tuple[str, str, int, int]] = []
    correct = 0

    for s in states:
        can_split = len(s.player_cards) == 2 and s.player_cards[0].rank_value == s.player_cards[1].rank_value
        can_double = len(s.player_cards) == 2
        obs = env.obs_from_cards(
            player_cards=s.player_cards,
            dealer_upcard=s.dealer_upcard,
            force_can_split=can_split,
            force_can_double=can_double,
            remaining_hands=0,
            is_first_decision=True,
            phase_is_bet=False,
        )
        mask = _build_play_mask(env, s.player_cards)

        kwargs = {"deterministic": True}
        if using_maskable:
            kwargs["action_masks"] = mask
        pred, _ = model.predict(obs, **kwargs)
        pred = int(pred)
        if pred > 3:
            pred = 0

        oracle = oracle_action(s.player_cards, s.dealer_upcard.rank, can_double=can_double, can_split=can_split)
        if pred == oracle:
            correct += 1
        else:
            mismatches.append((s.label, s.dealer_upcard.rank, pred, oracle))

    total = len(states)
    acc = 100.0 * correct / max(1, total)
    print(f"Canonical states evaluated: {total}")
    print(f"PLAY action accuracy vs oracle: {acc:.2f}%")

    if mismatches:
        print(f"Top mismatches ({min(args.show_mismatches, len(mismatches))}):")
        for label, dealer, pred, oracle in mismatches[: args.show_mismatches]:
            print(f"  {label} vs {dealer}: predicted={pred}, oracle={oracle}")


if __name__ == "__main__":
    main()
