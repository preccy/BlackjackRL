from __future__ import annotations

import sys
from pathlib import Path

import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blackjack_env import BlackjackEnv


def _mask_fn(env):
    return env.action_masks()


def test_maskable_recurrent_ppo_shoe_mode_smoke() -> None:
    sb3 = pytest.importorskip("sb3_contrib")
    if not hasattr(sb3, "MaskableRecurrentPPO"):
        pytest.skip("Installed sb3-contrib does not provide MaskableRecurrentPPO")

    maskable_recurrent_cls = sb3.MaskableRecurrentPPO
    action_masker_cls = sb3.common.wrappers.ActionMasker

    def make_env():
        env = BlackjackEnv(
            seed=11,
            obs_version=3,
            episode_mode="shoe",
            max_rounds_per_episode=50,
        )
        env = action_masker_cls(env, _mask_fn)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    model = maskable_recurrent_cls(
        "MlpLstmPolicy",
        vec_env,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        device="cpu",
        seed=17,
    )

    model.learn(total_timesteps=256)

    obs = vec_env.reset()
    state = None
    episode_start = [True]
    resolved_rounds = 0

    for _ in range(200):
        mask = vec_env.env_method("action_masks")[0]
        action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=True,
            action_masks=mask,
        )
        obs, _reward, dones, infos = vec_env.step(action)
        info = infos[0]
        if info.get("round_end", bool(info.get("outcomes"))):
            resolved_rounds += 1
        if dones[0]:
            episode_start = [True]
            state = None
        else:
            episode_start = [False]
        if resolved_rounds >= 10:
            break

    vec_env.close()

    assert resolved_rounds > 0
