from __future__ import annotations

from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


class TrainingEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn: Callable,
        eval_freq: int,
        n_eval_hands: int,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_hands = max(1, int(n_eval_hands))
        self.deterministic = deterministic
        self._mask_warning_printed = False

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        self._run_evaluation()
        return True

    @staticmethod
    def _get_action_mask(env):
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

    @staticmethod
    def _is_maskable_model(model) -> bool:
        try:
            from sb3_contrib import MaskablePPO

            return isinstance(model, MaskablePPO)
        except Exception:
            return model.__class__.__name__ == "MaskablePPO"

    def _run_evaluation(self) -> None:
        eval_env = self.eval_env_fn()
        using_maskable = self._is_maskable_model(self.model)

        wins = 0
        losses = 0
        pushes = 0
        resolved_rounds = 0
        resolved_hands_total = 0
        total_reward = 0.0

        def process_round_info(round_info):
            nonlocal resolved_rounds, resolved_hands_total, wins, pushes, losses
            outcomes = round_info.get("outcomes") or getattr(eval_env, "last_info", {}).get("outcomes", [])
            if not round_info.get("round_end", bool(outcomes)):
                return
            resolved_rounds += 1
            resolved_hands_total += len(outcomes)
            for outcome in outcomes:
                r = outcome.get("reward", 0)
                if r > 0:
                    wins += 1
                elif r == 0:
                    pushes += 1
                else:
                    losses += 1

        try:
            obs = None
            info = {}
            reset_seed = 0
            while resolved_rounds < self.n_eval_hands:
                if obs is None:
                    obs, info = eval_env.reset(seed=reset_seed)
                    reset_seed += 1
                    process_round_info(info)

                if getattr(eval_env, "terminated", False) or "immediate_reward" in info:
                    mask = self._get_action_mask(eval_env)
                    dummy_action = 0
                    if using_maskable and mask is not None:
                        action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
                        dummy_action = int(action)
                    obs, reward, terminated, truncated, info = eval_env.step(dummy_action)
                    total_reward += reward
                    if terminated or truncated:
                        process_round_info(info)
                        obs = None
                    continue

                if using_maskable:
                    mask = self._get_action_mask(eval_env)
                    if mask is None and not self._mask_warning_printed:
                        print("[TRAIN EVAL] Warning: action masks unavailable; continuing without masks.")
                        self._mask_warning_printed = True

                    if mask is not None:
                        action, _ = self.model.predict(
                            obs,
                            deterministic=self.deterministic,
                            action_masks=mask,
                        )
                    else:
                        action, _ = self.model.predict(obs, deterministic=self.deterministic)
                else:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)

                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                total_reward += reward
                process_round_info(info)

                if terminated or truncated:
                    obs = None
        finally:
            eval_env.close()

        denom_rounds = max(1, resolved_rounds)
        denom_hands = max(1, resolved_hands_total)
        ev = total_reward / denom_rounds
        win_rate = wins / denom_hands

        print("------------------------------------------------")
        print("[TRAIN EVAL]")
        print(f"Step: {self.num_timesteps}")
        print(f"Rounds: {resolved_rounds}")
        print(f"Hands: {resolved_hands_total}")
        print(f"EV: {ev:+.3f}")
        print(f"Win rate: {win_rate * 100:.1f}%")
        print(f"Net units: {total_reward:+.0f}")
        print("------------------------------------------------")
