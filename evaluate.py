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


def get_action_mask(env: BlackjackEnv):
    if hasattr(env, "action_masks"):
        return env.action_masks()
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "action_masks"):
        return unwrapped.action_masks()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/blackjack_ppo.zip")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--save-replays", type=int, default=50)
    parser.add_argument("--replay-out", type=str, default="./replays/eval_bundle.json")
    parser.add_argument("--algo", choices=["auto", "maskable", "ppo"], default="auto")
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

    env = BlackjackEnv(record_events=True)

    mask_warning_printed = False
    mask_aware_predict = using_maskable or args.algo == "maskable"
    print(f"Evaluation using mask-aware predict: {mask_aware_predict}")

    episode_rewards = []
    episode_count = 0
    resolved_hands = 0
    wins = pushes = losses = blackjacks = 0
    interesting = []

    for ep in range(args.hands):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        final_info = info
        while not done:
            if using_maskable:
                mask = get_action_mask(env)
                if mask is None and not mask_warning_printed:
                    print("Warning: action masks unavailable during evaluation; continuing without masks.")
                    mask_warning_printed = True
                if mask is not None:
                    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, r, term, trunc, info = env.step(int(action))
            ep_reward += r
            done = term or trunc
            final_info = info

        episode_count += 1
        episode_rewards.append(ep_reward)
        outcomes = final_info.get("outcomes", [])
        resolved_hands += len(outcomes)
        for out in outcomes:
            if out["reward"] > 0:
                wins += 1
            elif out["reward"] == 0:
                pushes += 1
            else:
                losses += 1
            if out["outcome"] in {"win_blackjack", "push_blackjack"}:
                blackjacks += 1

        if len(interesting) < args.save_replays and pick_interesting(final_info):
            interesting.append(env.export_episode())

    denom_hands = max(1, resolved_hands)
    print(f"Episodes played: {episode_count}")
    print(f"Resolved hands: {resolved_hands}")
    print(f"EV per round (episode): {np.mean(episode_rewards):.5f}")
    print(f"Win rate (resolved hands): {wins / denom_hands:.5f}")
    print(f"Push rate (resolved hands): {pushes / denom_hands:.5f}")
    print(f"Loss rate (resolved hands): {losses / denom_hands:.5f}")
    print(f"Blackjack rate (resolved hands): {blackjacks / denom_hands:.5f}")

    if interesting:
        Path(args.replay_out).parent.mkdir(parents=True, exist_ok=True)
        save_replay_bundle(args.replay_out, interesting)
        print(f"Saved {len(interesting)} replays to {args.replay_out}")


if __name__ == "__main__":
    main()
