from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from blackjack_env import BlackjackEnv


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
    parser.add_argument("--tensorboard-log", type=str, default="./tb_logs")
    parser.add_argument("--model-out", type=str, default="./models/blackjack_ppo")
    parser.add_argument("--model-in", type=str, default=None, help="Optional checkpoint to continue training from.")
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
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        print(f"Set torch num threads: {args.torch_threads}")

    device = resolve_device(args.device)
    print(f"Selected device: {device}")
    print(f"Vector env setup: base_seed={args.seed}, n_envs={args.n_envs}")


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
        env = DummyVecEnv(env_fns)
        algo_cls = MaskablePPO
        print("Training algorithm: MaskablePPO (action masking enabled)")
    else:
        env_fns = [make_env(rank, args.seed) for rank in range(args.n_envs)]
        env = DummyVecEnv(env_fns)
        algo_cls = PPO
        print("Training algorithm: PPO (no action-mask support)")

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

    model.learn(total_timesteps=args.total_timesteps)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}.zip")


if __name__ == "__main__":
    main()
