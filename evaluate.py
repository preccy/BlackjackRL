from __future__ import annotations

import argparse
import time
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

    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        try:
            return get_wrapper_attr("action_masks")()
        except Exception:
            pass

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "action_masks"):
        return unwrapped.action_masks()

    current = getattr(env, "env", None)
    visited = {id(env)}
    while current is not None and id(current) not in visited:
        if hasattr(current, "action_masks"):
            return current.action_masks()
        visited.add(id(current))
        current = getattr(current, "env", None)

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/blackjack_ppo.zip")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--save-replays", type=int, default=50)
    parser.add_argument("--replay-out", type=str, default="./replays/eval_bundle.json")
    parser.add_argument("--algo", choices=["auto", "maskable", "ppo"], default="auto")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--progress-update-every", type=int, default=250)
    parser.set_defaults(progress=True)
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

    progress = None
    if args.progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=args.hands, unit="hand", dynamic_ncols=True, desc="Evaluating")
        except Exception as exc:
            print(f"Warning: could not initialize tqdm progress bar ({exc!r}); continuing without it.")

    start = time.perf_counter()
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

        if progress is not None:
            progress.update(1)
            if (ep + 1) % max(1, args.progress_update_every) == 0 or (ep + 1) == args.hands:
                elapsed = max(1e-9, time.perf_counter() - start)
                denom_hands = max(1, resolved_hands)
                progress.set_postfix(
                    resolved=resolved_hands,
                    ev=f"{np.mean(episode_rewards):.4f}",
                    win_rate=f"{wins / denom_hands:.3f}",
                    net=f"{np.sum(episode_rewards):.1f}",
                    hps=f"{episode_count / elapsed:.1f}",
                )

    if progress is not None:
        progress.close()

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
