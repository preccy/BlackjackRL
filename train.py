from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from blackjack_env import BlackjackEnv
from callbacks.eval_callback import TrainingEvalCallback
from imitation_pretrain import parse_bet_levels, run_basic_strategy_pretrain, validate_pretrain_config


ActionMasker = None


def mask_fn(env):
    attempts = []

    if hasattr(env, "action_masks"):
        attempts.append("env.action_masks")
        return env.action_masks()

    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        attempts.append("env.get_wrapper_attr('action_masks')")
        try:
            return get_wrapper_attr("action_masks")()
        except Exception as exc:
            attempts.append(f"get_wrapper_attr lookup failed: {exc!r}")

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "action_masks"):
        attempts.append("env.unwrapped.action_masks")
        return unwrapped.action_masks()

    attempts.append("walk .env wrapper chain")
    chain = [type(env).__name__]
    current = getattr(env, "env", None)
    visited = {id(env)}
    while current is not None and id(current) not in visited:
        chain.append(type(current).__name__)
        if hasattr(current, "action_masks"):
            return current.action_masks()
        visited.add(id(current))
        current = getattr(current, "env", None)

    raise RuntimeError(
        "Unable to resolve action_masks() from wrapped environment. "
        f"Wrapper chain: {' -> '.join(chain)}. "
        f"Lookup attempts: {', '.join(attempts)}. "
        "Ensure ActionMasker wraps the base environment before Monitor "
        "(expected order: Monitor(ActionMasker(base_env)))."
    )


def make_env(rank: int, base_seed: int, obs_version: int, episode_mode: str, max_rounds_per_episode: int, enable_betting: bool, bet_levels: list[float], bankroll_start: float | None, bankroll_stop_on_zero: bool, terminate_on_bankroll_bust: bool, betting_reward_mode: str, bet_exploration_bonus: float, bet_exploration_mode: str):
    def _init():
        env = BlackjackEnv(
            seed=base_seed + rank,
            obs_version=obs_version,
            episode_mode=episode_mode,
            max_rounds_per_episode=max_rounds_per_episode,
            enable_betting=enable_betting,
            bet_levels=bet_levels,
            bankroll_start=bankroll_start,
            bankroll_stop_on_zero=bankroll_stop_on_zero,
            terminate_on_bankroll_bust=terminate_on_bankroll_bust,
            betting_reward_mode=betting_reward_mode,
            bet_exploration_bonus=bet_exploration_bonus,
            bet_exploration_mode=bet_exploration_mode,
        )
        if ActionMasker is not None:
            env = ActionMasker(env, mask_fn)
        return Monitor(env)

    return _init


def parse_cpu_affinity(cpu_affinity: str | None) -> list[int] | None:
    if not cpu_affinity:
        return None
    cpus = []
    for token in cpu_affinity.split(","):
        token = token.strip()
        if not token:
            continue
        cpus.append(int(token))
    if not cpus:
        return None
    return cpus


def apply_cpu_affinity(cpu_list: list[int] | None) -> str:
    if not cpu_list:
        return "not requested"
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpu_list))
            return f"applied via os.sched_setaffinity: {cpu_list}"

        import psutil  # type: ignore

        proc = psutil.Process()
        proc.cpu_affinity(cpu_list)
        return f"applied via psutil: {cpu_list}"
    except Exception as exc:
        return f"failed to apply ({exc!r})"


def configure_cpu_runtime(args, device: str) -> None:
    if device != "cpu":
        return

    if args.torch_threads is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))
        torch.set_num_threads(args.torch_threads)

    if args.torch_interop_threads is not None:
        try:
            torch.set_num_interop_threads(args.torch_interop_threads)
        except RuntimeError as exc:
            print(f"Warning: unable to set torch interop threads ({exc!r}).")


def build_vec_env(n_envs: int, env_fns, vec_env: str):
    vec_env_mode = vec_env
    if vec_env_mode == "auto":
        vec_env_mode = "dummy" if n_envs == 1 else "subproc"

    if vec_env_mode == "dummy":
        return DummyVecEnv(env_fns), "DummyVecEnv"

    start_method = "spawn" if platform.system().lower().startswith("win") else "forkserver"
    return SubprocVecEnv(env_fns, start_method=start_method), f"SubprocVecEnv(start_method={start_method})"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "auto"
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: --device cuda requested but CUDA is unavailable; falling back to cpu.")
        return "cpu"
    return requested


def _save_meta(model_out: str, obs_version: int, episode_mode: str, max_rounds_per_episode: int, enable_betting: bool, bet_levels: list[float], bankroll_start: float | None, bankroll_stop_on_zero: bool, terminate_on_bankroll_bust: bool, betting_reward_mode: str, bet_exploration_bonus: float, bet_exploration_mode: str) -> None:
    model_path = Path(model_out)
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json") if model_path.suffix else Path(f"{model_out}.meta.json")
    meta = {
        "env": {
            "n_decks": 6,
            "penetration": 0.25,
            "dealer_stands_soft17": True,
            "blackjack_payout": 1.5,
            "das": True,
            "max_hands": 4,
            "illegal_action_penalty": -0.05,
            "obs_version": obs_version,
            "episode_mode": episode_mode,
            "max_rounds_per_episode": max_rounds_per_episode,
            "enable_betting": enable_betting,
            "bet_levels": bet_levels,
            "bankroll_start": bankroll_start,
            "bankroll_stop_on_zero": bankroll_stop_on_zero,
            "terminate_on_bankroll_bust": terminate_on_bankroll_bust,
            "betting_reward_mode": betting_reward_mode,
            "bet_exploration_bonus": bet_exploration_bonus,
            "bet_exploration_mode": bet_exploration_mode,
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata to {meta_path}")


def main() -> None:
    global ActionMasker

    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", "--timesteps", dest="total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="Base random seed; each worker uses seed+rank.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto")
    parser.add_argument("--cpu-affinity", type=str, default=None, help="Optional CPU list, e.g. 0,1,2,3,4,5")
    parser.add_argument("--tensorboard-log", type=str, default="./tb_logs")
    parser.add_argument("--model-out", type=str, default="./models/blackjack_ppo")
    parser.add_argument("--model-in", type=str, default=None, help="Optional checkpoint to continue training from.")
    parser.add_argument("--train-eval-freq", type=int, default=100_000)
    parser.add_argument("--train-eval-hands", type=int, default=20_000)
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--episode-mode", choices=["hand", "shoe"], default="hand")
    parser.add_argument("--max-rounds-per-episode", type=int, default=200)
    parser.add_argument("--enable-betting", action="store_true")
    parser.add_argument("--bet-levels", type=str, default="1")
    parser.add_argument("--bankroll-start", type=float, default=None)
    parser.add_argument("--bankroll-stop-on-zero", action="store_true")
    parser.add_argument("--terminate-on-bankroll-bust", dest="terminate_on_bankroll_bust", action="store_true")
    parser.add_argument("--no-terminate-on-bankroll-bust", dest="terminate_on_bankroll_bust", action="store_false")
    parser.add_argument("--betting-reward-mode", choices=["net", "roi", "log_bankroll"], default="net")
    parser.add_argument("--bet-exploration-bonus", type=float, default=0.0)
    parser.add_argument("--bet-exploration-mode", choices=["none", "scaled_index"], default="none")
    parser.add_argument("--pretrain-basic-strategy", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-samples", type=int, default=200_000)
    parser.add_argument("--pretrain-bet-mode", choices=["minbet"], default="minbet")
    parser.add_argument(
        "--masking",
        action="store_true",
        help="Require MaskablePPO training (fails if sb3-contrib is unavailable unless --allow-unmasked-fallback is set).",
    )
    parser.add_argument(
        "--allow-unmasked-fallback",
        action="store_true",
        help="Allow fallback to vanilla PPO when --masking is set but MaskablePPO is unavailable.",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="Use MaskableRecurrentPPO with MlpLstmPolicy (requires sb3-contrib maskable support).",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True, terminate_on_bankroll_bust=None)
    args = parser.parse_args()

    if args.terminate_on_bankroll_bust is None:
        args.terminate_on_bankroll_bust = args.bankroll_start is not None

    bet_levels = parse_bet_levels(args.bet_levels)

    if args.pretrain_basic_strategy:
        validate_pretrain_config(args.obs_version, args.enable_betting, bet_levels)

    device = resolve_device(args.device)
    configure_cpu_runtime(args, device)
    affinity_status = apply_cpu_affinity(parse_cpu_affinity(args.cpu_affinity))

    maskable_available = False
    maskable_error = None
    recurrent_maskable_error = None
    MaskablePPO = None
    MaskableRecurrentPPO = None
    ActionMasker = None
    try:
        from sb3_contrib import MaskablePPO as _MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker as _ActionMasker

        MaskablePPO = _MaskablePPO
        ActionMasker = _ActionMasker
        maskable_available = True
    except Exception as exc:
        maskable_error = exc
        print("MaskablePPO unavailable, falling back to vanilla PPO (illegal action fallback/penalty behavior applies).")

    if maskable_available:
        try:
            from sb3_contrib import MaskableRecurrentPPO as _MaskableRecurrentPPO

            MaskableRecurrentPPO = _MaskableRecurrentPPO
        except Exception as exc:
            recurrent_maskable_error = exc
            print("MaskableRecurrentPPO unavailable; --recurrent cannot be used in this environment.")

    if args.masking and not maskable_available and not args.allow_unmasked_fallback:
        raise RuntimeError(
            "--masking was requested but MaskablePPO could not be imported. "
            "Install sb3-contrib or pass --allow-unmasked-fallback to continue with PPO."
        ) from maskable_error

    if args.recurrent and MaskableRecurrentPPO is None:
        raise RuntimeError(
            "--recurrent was requested but MaskableRecurrentPPO could not be imported. "
            "Install/upgrade sb3-contrib to a version that includes MaskableRecurrentPPO."
        ) from (recurrent_maskable_error or maskable_error)

    use_maskable = maskable_available
    if args.masking:
        use_maskable = maskable_available
    elif not maskable_available:
        use_maskable = False

    env_fns = [
        make_env(
            rank,
            args.seed,
            args.obs_version,
            args.episode_mode,
            args.max_rounds_per_episode,
            args.enable_betting,
            bet_levels,
            args.bankroll_start,
            args.bankroll_stop_on_zero,
            args.terminate_on_bankroll_bust,
            args.betting_reward_mode,
            args.bet_exploration_bonus,
            args.bet_exploration_mode,
        )
        for rank in range(args.n_envs)
    ]
    env, actual_vec_env = build_vec_env(args.n_envs, env_fns, args.vec_env)
    if args.recurrent:
        algo_cls = MaskableRecurrentPPO
    else:
        algo_cls = MaskablePPO if use_maskable else PPO
    masking_enabled = use_maskable

    print("=== Training startup configuration ===")
    print(f"Selected device: {device}")
    print(f"Algorithm: {algo_cls.__name__}")
    print(f"Masking enabled: {masking_enabled}")
    print(f"Observation version: {args.obs_version}")
    print(f"Episode mode: {args.episode_mode} (max_rounds_per_episode={args.max_rounds_per_episode})")
    print(f"Betting enabled: {args.enable_betting} (bet_levels={bet_levels})")
    print(
        f"Bet reward mode: {args.betting_reward_mode} "
        f"(exploration mode={args.bet_exploration_mode}, exploration bonus={args.bet_exploration_bonus})"
    )
    print(f"Vector env setup: base_seed={args.seed}, n_envs={args.n_envs}, vec_env={actual_vec_env}")
    print(f"torch num threads: {torch.get_num_threads()}")
    print(f"torch interop threads: {torch.get_num_interop_threads()}")
    print(
        "CPU thread env vars: "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}, "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}"
    )
    print(f"CPU affinity: {affinity_status}")
    print(f"Progress bar: {'enabled' if args.progress else 'disabled'}")

    model_kwargs = dict(
        policy="MlpLstmPolicy" if args.recurrent else "MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        n_steps=4096,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        vf_coef=0.5,
        n_epochs=10,
        device=device,
    )

    if args.model_in:
        print(f"Loading model from {args.model_in}")
        model = algo_cls.load(args.model_in, env=env, device=device)
    else:
        model = algo_cls(**model_kwargs)

    if args.pretrain_basic_strategy:
        pretrain_env = BlackjackEnv(
            seed=args.seed,
            obs_version=args.obs_version,
            episode_mode="hand",
            enable_betting=args.enable_betting,
            bet_levels=bet_levels,
            betting_reward_mode=args.betting_reward_mode,
            bet_exploration_bonus=args.bet_exploration_bonus,
            bet_exploration_mode=args.bet_exploration_mode,
        )
        stats = run_basic_strategy_pretrain(
            model,
            pretrain_env,
            epochs=args.pretrain_epochs,
            samples=args.pretrain_samples,
            seed=args.seed,
            enable_betting=args.enable_betting,
            bet_levels=bet_levels,
            pretrain_bet_mode=args.pretrain_bet_mode,
        )
        print(
            f"Pretraining complete: samples={stats.samples}, epochs={stats.epochs}, "
            f"final_loss={stats.final_loss:.4f}, best_canonical_accuracy={100.0 * stats.best_canonical_accuracy:.2f}%"
        )

    eval_env_seed_base = args.seed + 100_000

    def make_eval_env():
        return BlackjackEnv(
            seed=eval_env_seed_base,
            record_events=False,
            obs_version=args.obs_version,
            episode_mode=args.episode_mode,
            max_rounds_per_episode=args.max_rounds_per_episode,
            enable_betting=args.enable_betting,
            bet_levels=bet_levels,
            bankroll_start=args.bankroll_start,
            bankroll_stop_on_zero=args.bankroll_stop_on_zero,
            terminate_on_bankroll_bust=args.terminate_on_bankroll_bust,
            betting_reward_mode=args.betting_reward_mode,
            bet_exploration_bonus=args.bet_exploration_bonus,
            bet_exploration_mode=args.bet_exploration_mode,
        )

    callback = TrainingEvalCallback(
        eval_env_fn=make_eval_env,
        eval_freq=args.train_eval_freq,
        n_eval_hands=args.train_eval_hands,
    )

    start = time.perf_counter()
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=args.progress)
    except ImportError as exc:
        if args.progress:
            print(
                "Warning: progress bar dependencies missing (install `tqdm` and `rich`). "
                f"Continuing without progress bar. Details: {exc!r}"
            )
            model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
        else:
            raise
    elapsed = max(1e-9, time.perf_counter() - start)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    _save_meta(
        args.model_out,
        obs_version=args.obs_version,
        episode_mode=args.episode_mode,
        max_rounds_per_episode=args.max_rounds_per_episode,
        enable_betting=args.enable_betting,
        bet_levels=bet_levels,
        bankroll_start=args.bankroll_start,
        bankroll_stop_on_zero=args.bankroll_stop_on_zero,
        terminate_on_bankroll_bust=args.terminate_on_bankroll_bust,
        betting_reward_mode=args.betting_reward_mode,
        bet_exploration_bonus=args.bet_exploration_bonus,
        bet_exploration_mode=args.bet_exploration_mode,
    )
    print(f"Saved model to {args.model_out}.zip")
    print(f"Total timesteps seen: {model.num_timesteps}")
    print(f"Elapsed wall time: {elapsed:.2f}s")
    print(f"Approx throughput: {model.num_timesteps / elapsed:.2f} timesteps/s")


if __name__ == "__main__":
    main()
