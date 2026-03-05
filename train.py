from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from blackjack_env import BlackjackEnv


def mask_fn(env):
    return env.action_masks()


def make_env(seed_offset: int = 0):
    def _init():
        env = BlackjackEnv(seed=seed_offset)
        return Monitor(env)

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--tensorboard-log", type=str, default="./tb_logs")
    parser.add_argument("--model-out", type=str, default="./models/blackjack_ppo")
    args = parser.parse_args()

    vec_env = make_vec_env(make_env(), n_envs=args.n_envs)

    model = None
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker

        wrapped_env = make_vec_env(lambda: ActionMasker(make_env()(), mask_fn), n_envs=args.n_envs)
        model = MaskablePPO(
            "MlpPolicy",
            wrapped_env,
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
        )
    except Exception:
        model = PPO(
            "MlpPolicy",
            vec_env,
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
        )

    model.learn(total_timesteps=args.total_timesteps)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}.zip")


if __name__ == "__main__":
    main()
