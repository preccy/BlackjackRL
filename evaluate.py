from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from blackjack_env import BlackjackEnv
from replay_logger import save_replay_bundle


def pick_interesting(info: dict) -> bool:
    outcomes = info.get("outcomes", [])
    labels = {o["outcome"] for o in outcomes}
    return any(
        key in labels
        for key in ["win_blackjack", "push_blackjack", "loss_dealer_blackjack", "win_dealer_bust"]
    ) or len(outcomes) > 1 or any(abs(o["reward"]) >= 2 for o in outcomes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/blackjack_ppo.zip")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--save-replays", type=int, default=50)
    parser.add_argument("--replay-out", type=str, default="./replays/eval_bundle.json")
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(args.model)
    except Exception:
        model = PPO.load(args.model)
    env = BlackjackEnv(record_events=True)

    rewards = []
    wins = pushes = blackjacks = 0
    interesting = []

    for ep in range(args.hands):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        final_info = info
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ep_reward += r
            done = term or trunc
            final_info = info

        rewards.append(ep_reward)
        outcomes = final_info.get("outcomes", [])
        for out in outcomes:
            if out["reward"] > 0:
                wins += 1
            elif out["reward"] == 0:
                pushes += 1
            if out["outcome"] in {"win_blackjack", "push_blackjack"}:
                blackjacks += 1

        if len(interesting) < args.save_replays and pick_interesting(final_info):
            interesting.append(env.export_episode())

    total = max(1, sum(len(r.get("info", {}).get("outcomes", [])) for r in interesting))
    print(f"Hands: {args.hands}")
    print(f"EV per hand: {np.mean(rewards):.5f}")
    print(f"Win rate: {wins / args.hands:.5f}")
    print(f"Push rate: {pushes / args.hands:.5f}")
    print(f"Blackjack rate: {blackjacks / args.hands:.5f}")

    if interesting:
        Path(args.replay_out).parent.mkdir(parents=True, exist_ok=True)
        save_replay_bundle(args.replay_out, interesting)
        print(f"Saved {len(interesting)} replays to {args.replay_out}")


if __name__ == "__main__":
    main()
