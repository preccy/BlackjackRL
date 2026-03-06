from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from stable_baselines3 import PPO

from blackjack_env import BlackjackEnv, HandState
from utils.cards import Card


ACTION_SYMBOLS = {
    0: "S",  # stand
    1: "H",  # hit
    2: "D",  # double
    3: "P",  # split
}

DEALER_COLUMNS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]


@dataclass
class LoadedModel:
    model: object
    algo_name: str
    mask_aware: bool


def _card(rank: str, suit: str = "♠") -> Card:
    return Card(rank=rank, suit=suit)


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
            return int(meta.get("env", {}).get("obs_version", 1))
        except Exception as exc:
            print(f"Warning: could not parse metadata {meta_path} ({exc!r}); defaulting to obs_version=1")
            break
    return 1


def load_model(model_path: str, algo: str) -> LoadedModel:
    if algo in {"auto", "maskable", "recurrent"}:
        try:
            if algo == "recurrent":
                from sb3_contrib import MaskableRecurrentPPO

                return LoadedModel(
                    model=MaskableRecurrentPPO.load(model_path),
                    algo_name="MaskableRecurrentPPO",
                    mask_aware=True,
                )

            from sb3_contrib import MaskablePPO

            return LoadedModel(model=MaskablePPO.load(model_path), algo_name="MaskablePPO", mask_aware=True)
        except Exception:
            if algo != "auto":
                raise

    return LoadedModel(model=PPO.load(model_path), algo_name="PPO", mask_aware=False)


def set_manual_state(env: BlackjackEnv, player_cards: list[Card], dealer_upcard_rank: str) -> tuple:
    env.terminated = False
    env.current_hand_idx = 0
    env.hands = [HandState(cards=list(player_cards), done=False)]
    env.dealer_cards = [_card(dealer_upcard_rank, "♥")]
    obs = env._obs()
    mask = env.action_masks()
    return obs, mask


def predict_action_symbol(model_info: LoadedModel, obs, mask) -> str:
    if model_info.mask_aware:
        action, _ = model_info.model.predict(obs, deterministic=True, action_masks=mask)
    else:
        action, _ = model_info.model.predict(obs, deterministic=True)
    return ACTION_SYMBOLS[int(action)]


def hard_total_cards(total: int) -> list[Card]:
    if total == 8:
        ranks = ["6", "2"]
    elif total == 9:
        ranks = ["7", "2"]
    elif total == 10:
        ranks = ["8", "2"]
    elif total == 11:
        ranks = ["9", "2"]
    elif 12 <= total <= 19:
        ranks = ["T", str(total - 10)]
    elif total == 20:
        ranks = ["T", "7", "3"]
    else:
        raise ValueError(f"Unsupported hard total: {total}")
    return [_card(r, "♠") for r in ranks]


def soft_hand_cards(kicker: int) -> list[Card]:
    return [_card("A", "♠"), _card(str(kicker), "♦")]


def pair_cards(pair_rank: str) -> list[Card]:
    return [_card(pair_rank, "♠"), _card(pair_rank, "♦")]


def build_table(env: BlackjackEnv, model_info: LoadedModel, row_specs: Iterable[tuple[str, list[Card]]]) -> list[list[str]]:
    table = []
    for row_label, cards in row_specs:
        actions = []
        for dealer in DEALER_COLUMNS:
            obs, mask = set_manual_state(env, cards, dealer)
            actions.append(predict_action_symbol(model_info, obs, mask))
        table.append([row_label, *actions])
    return table


def print_table(title: str, rows: list[list[str]]) -> None:
    print("=" * 30)
    print(f"{title} STRATEGY")
    print("=" * 30)
    print()
    print("      " + " ".join(DEALER_COLUMNS))
    for row in rows:
        label, *cells = row
        print(f"{label:<5} " + " ".join(cells))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print learned blackjack strategy tables from a trained RL model.")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained model .zip")
    parser.add_argument(
        "--algo",
        choices=["auto", "maskable", "recurrent", "ppo"],
        default="auto",
        help="Model algorithm type (maskable supports MaskablePPO, recurrent for MaskableRecurrentPPO).",
    )
    parser.add_argument("--obs-version", type=int, choices=[1, 2], default=None)
    args = parser.parse_args()

    model_info = load_model(args.model, args.algo)
    obs_version = _resolve_obs_version(args.model, args.obs_version)
    print(f"Loaded model with {model_info.algo_name}")
    print(f"Using obs_version={obs_version}")

    env = BlackjackEnv(seed=0, n_decks=1, penetration=0.0, obs_version=obs_version)

    hard_rows = [(str(total), hard_total_cards(total)) for total in range(8, 21)]
    soft_rows = [(f"A{k}", soft_hand_cards(k)) for k in range(2, 10)]
    pair_labels = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]
    pair_rows = [(f"{r}{r}", pair_cards(r)) for r in pair_labels]

    print_table("HARD TOTALS", build_table(env, model_info, hard_rows))
    print_table("SOFT HANDS", build_table(env, model_info, soft_rows))
    print_table("PAIRS", build_table(env, model_info, pair_rows))


if __name__ == "__main__":
    main()
