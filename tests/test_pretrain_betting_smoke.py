from __future__ import annotations

import sys
from pathlib import Path

import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from blackjack_env import BlackjackEnv
from imitation_pretrain import run_basic_strategy_pretrain


def _mask_fn(env):
    return env.action_masks()


def test_pretrain_betting_phase_smoke(tmp_path: Path) -> None:
    sb3 = pytest.importorskip("sb3_contrib")
    if not hasattr(sb3, "MaskablePPO"):
        pytest.skip("Installed sb3-contrib does not provide MaskablePPO")

    maskable_cls = sb3.MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    action_masker_cls = ActionMasker

    bet_levels = [1.0, 2.0, 4.0, 8.0]

    def make_env():
        env = BlackjackEnv(
            seed=31,
            obs_version=4,
            episode_mode="shoe",
            max_rounds_per_episode=50,
            enable_betting=True,
            bet_levels=bet_levels,
            bankroll_start=200.0,
        )
        env = action_masker_cls(env, _mask_fn)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    model = maskable_cls(
        "MlpPolicy",
        vec_env,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        device="cpu",
        seed=33,
    )

    pretrain_env = BlackjackEnv(
        seed=33,
        obs_version=4,
        episode_mode="hand",
        enable_betting=True,
        bet_levels=bet_levels,
    )
    stats = run_basic_strategy_pretrain(
        model,
        pretrain_env,
        epochs=2,
        samples=2_000,
        seed=33,
        enable_betting=True,
        bet_levels=bet_levels,
    )
    assert stats.samples == 2_000

    model_path = tmp_path / "pretrain_bet_model"
    model.save(str(model_path))
    loaded_model = maskable_cls.load(str(model_path) + ".zip", env=vec_env, device="cpu")

    eval_env = BlackjackEnv(
        seed=45,
        obs_version=4,
        episode_mode="shoe",
        max_rounds_per_episode=50,
        enable_betting=True,
        bet_levels=bet_levels,
        bankroll_start=200.0,
    )

    for i in range(4):
        obs, info = eval_env.reset(seed=100 + i)
        assert eval_env.phase == "BET"
        bet_mask = info["action_mask"]
        assert bet_mask[:4].sum() == 0
        assert bet_mask[4:].all()

        bet_action, _ = loaded_model.predict(obs, deterministic=True, action_masks=bet_mask)
        assert int(bet_action) >= 4

        obs, _reward, _term, _trunc, info = eval_env.step(int(bet_action))
        assert eval_env.phase == "PLAY"
        play_mask = info["action_mask"]
        assert play_mask[:4].sum() >= 1
        assert play_mask[4:].sum() == 0

        play_action, _ = loaded_model.predict(obs, deterministic=True, action_masks=play_mask)
        assert 0 <= int(play_action) <= 3

    vec_env.close()
