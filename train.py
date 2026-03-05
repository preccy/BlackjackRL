from __future__ import annotations

import argparse
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


def make_env(rank: int, base_seed: int):
    def _init():
        env = BlackjackEnv(seed=base_seed + rank)
        # Keep Monitor statistics while ensuring mask_fn receives the base env.
        # This wrapper order avoids ActionMasker passing a Monitor into mask_fn.
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
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    configure_cpu_runtime(args, device)
    affinity_status = apply_cpu_affinity(parse_cpu_affinity(args.cpu_affinity))


    maskable_available = False
    maskable_error = None
    MaskablePPO = None
    ActionMasker = None
    try:
        from sb3_contrib import MaskablePPO as _MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker as _ActionMasker

        MaskablePPO = _MaskablePPO
        ActionMasker = _ActionMasker
        maskable_available = True
    except Exception as exc:
        maskable_error = exc
        print(
            "MaskablePPO unavailable, falling back to vanilla PPO "
            "(illegal action fallback/penalty behavior applies)."
        )

    if args.masking and not maskable_available and not args.allow_unmasked_fallback:
        raise RuntimeError(
            "--masking was requested but MaskablePPO could not be imported. "
            "Install sb3-contrib or pass --allow-unmasked-fallback to continue with PPO."
        ) from maskable_error

    use_maskable = maskable_available
    if args.masking:
        use_maskable = maskable_available
    elif not maskable_available:
        use_maskable = False

    if use_maskable:
        env_fns = [make_env(rank, args.seed) for rank in range(args.n_envs)]
        env, actual_vec_env = build_vec_env(args.n_envs, env_fns, args.vec_env)
        algo_cls = MaskablePPO
        masking_enabled = True
    else:
        env_fns = [make_env(rank, args.seed) for rank in range(args.n_envs)]
        env, actual_vec_env = build_vec_env(args.n_envs, env_fns, args.vec_env)
        algo_cls = PPO
        masking_enabled = False

    print("=== Training startup configuration ===")
    print(f"Selected device: {device}")
    print(f"Algorithm: {algo_cls.__name__}")
    print(f"Masking enabled: {masking_enabled}")
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
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        n_steps=2048,
        batch_size=256,
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

    eval_env_seed_base = args.seed + 100_000

    def make_eval_env():
        return BlackjackEnv(seed=eval_env_seed_base, record_events=False)

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
    print(f"Saved model to {args.model_out}.zip")
    print(f"Total timesteps seen: {model.num_timesteps}")
    print(f"Elapsed wall time: {elapsed:.2f}s")
    print(f"Approx throughput: {model.num_timesteps / elapsed:.2f} timesteps/s")


if __name__ == "__main__":
    main()
