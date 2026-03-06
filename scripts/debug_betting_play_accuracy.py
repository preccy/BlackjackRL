from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blackjack_env import BlackjackEnv
from imitation_pretrain import evaluate_canonical_play_accuracy, parse_bet_levels


def _load_model(path: str):
    try:
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path, device="cpu")
    except Exception:
        from stable_baselines3 import PPO

        return PPO.load(path, device="cpu")


def _action_name(a: int) -> str:
    return {0: "STAND", 1: "HIT", 2: "DOUBLE", 3: "SPLIT"}.get(a, str(a))


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug PLAY policy accuracy against basic-strategy oracle.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--obs-version", type=int, default=4)
    ap.add_argument("--bet-levels", type=str, default="1,2,4,8")
    args = ap.parse_args()

    bet_levels = parse_bet_levels(args.bet_levels)
    env = BlackjackEnv(obs_version=args.obs_version, enable_betting=True, bet_levels=bet_levels, seed=7)
    model = _load_model(args.model)

    acc, rows, breakdown = evaluate_canonical_play_accuracy(model, env, include_breakdown=True)
    correct = sum(1 for _, tgt, pred, _ in rows if tgt == pred)
    total = len(rows)
    print(f"PLAY oracle accuracy: {correct}/{total} = {100.0 * acc:.2f}%")

    print("Category accuracy:")
    for category in ("hard", "soft", "pairs"):
        c, n = breakdown[category]
        pct = 100.0 * c / max(1, n)
        print(f"  {category:>5}: {c}/{n} = {pct:.2f}%")

    mismatches = [(lbl, t, p, cat) for lbl, t, p, cat in rows if t != p]
    if not mismatches:
        print("No mismatches.")
        return

    print("Top mismatches:")
    counts = Counter((t, p) for _, t, p, _ in mismatches)
    for (t, p), n in counts.most_common(12):
        print(f"  oracle={_action_name(t)} predicted={_action_name(p)} count={n}")

    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    for _, t, p, _ in rows:
        confusion[t][p] += 1

    print("Per-action confusion summary:")
    for oracle_action in (2, 3):
        total_t = sum(confusion[oracle_action].values())
        print(f"  oracle {_action_name(oracle_action)} ({total_t} samples):")
        for pred_action, n in confusion[oracle_action].most_common():
            pct = 100.0 * n / max(1, total_t)
            print(f"    -> predicted {_action_name(pred_action)}: {n} ({pct:.2f}%)")

    print("First mismatches:")
    for lbl, t, p, cat in mismatches[:20]:
        print(f"  [{cat}] {lbl}: oracle={_action_name(t)} predicted={_action_name(p)}")


if __name__ == "__main__":
    main()
