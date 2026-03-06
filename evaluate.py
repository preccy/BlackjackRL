from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from blackjack_env import BlackjackEnv
from replay_logger import save_replay_bundle


def pick_interesting(info: dict) -> bool:
    outcomes = info.get("outcomes") or info.get("info", {}).get("outcomes", [])
    labels = {o["outcome"] for o in outcomes}
    return any(
        key in labels
        for key in ["win_blackjack", "push_blackjack", "loss_dealer_blackjack", "win_dealer_bust"]
    ) or len(outcomes) > 1 or any(abs(o["reward"]) >= 2 for o in outcomes)


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
            obs_version = int(meta.get("env", {}).get("obs_version", 1))
            print(f"Using obs_version={obs_version} from {meta_path}")
            return obs_version
        except Exception as exc:
            print(f"Warning: could not parse metadata {meta_path} ({exc!r}); defaulting to obs_version=1")
            break
    return 1


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
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--episode-mode", choices=["hand", "shoe"], default="hand")
    parser.add_argument("--max-rounds-per-episode", type=int, default=200)
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

    obs_version = _resolve_obs_version(args.model, args.obs_version)
    env = BlackjackEnv(
        record_events=True,
        obs_version=obs_version,
        episode_mode=args.episode_mode,
        max_rounds_per_episode=args.max_rounds_per_episode,
    )

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
    resolved_rounds = 0
    resolved_hands_total = 0
    wins = pushes = losses = blackjacks = 0
    interesting = []

    def process_round_info(round_info: dict) -> bool:
        nonlocal resolved_rounds, resolved_hands_total, wins, pushes, losses, blackjacks, interesting
        outcomes = round_info.get("outcomes")
        round_finished = round_info.get("round_end", bool(outcomes))
        if not round_finished:
            return False
        outcomes = outcomes or getattr(env, "last_info", {}).get("outcomes", [])
        resolved_rounds += 1
        resolved_hands_total += len(outcomes)
        for out in outcomes:
            if out["reward"] > 0:
                wins += 1
            elif out["reward"] == 0:
                pushes += 1
            else:
                losses += 1
            if out["outcome"] in {"win_blackjack", "push_blackjack"}:
                blackjacks += 1
        if len(interesting) < args.save_replays and pick_interesting(round_info):
            interesting.append(round_info.get("round_replay") or env.export_episode())
        return True

    obs = None
    info = {}
    reset_seed = 0
    while resolved_rounds < args.hands:
        if obs is None:
            obs, info = env.reset(seed=reset_seed)
            reset_seed += 1
            episode_count += 1
            episode_rewards.append(0.0)
            process_round_info(info)

        if env.terminated or "immediate_reward" in info:
            mask = get_action_mask(env)
            dummy_action = 0
            if using_maskable and mask is not None:
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                dummy_action = int(action)
            obs, r, term, trunc, info = env.step(dummy_action)
            episode_rewards[-1] += r
            if term or trunc:
                process_round_info(info)
                obs = None
            continue

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
        episode_rewards[-1] += r

        round_finished = process_round_info(info)

        if term or trunc:
            obs = None

        if progress is not None and round_finished:
            progress.update(1)
            if resolved_rounds % max(1, args.progress_update_every) == 0 or resolved_rounds >= args.hands:
                elapsed = max(1e-9, time.perf_counter() - start)
                denom_rounds = max(1, resolved_rounds)
                denom_hands = max(1, resolved_hands_total)
                progress.set_postfix(
                    rounds=resolved_rounds,
                    hands=resolved_hands_total,
                    ev=f"{(np.sum(episode_rewards) / denom_rounds):.4f}",
                    win_rate=f"{wins / denom_hands:.3f}",
                    net=f"{np.sum(episode_rewards):.1f}",
                    rps=f"{resolved_rounds / elapsed:.1f}",
                )

    if progress is not None:
        progress.close()

    denom_rounds = max(1, resolved_rounds)
    denom_hands = max(1, resolved_hands_total)
    print(f"Episodes played: {episode_count}")
    print(f"Resolved rounds: {resolved_rounds}")
    print(f"Resolved hands: {resolved_hands_total}")
    total_net = float(np.sum(episode_rewards))
    print(f"EV per round: {total_net / denom_rounds:.5f}")
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
