# BlackjackRL

Blackjack reinforcement learning project with:

- Gymnasium-compatible `BlackjackEnv` implementing 6-deck blackjack with hit/stand/double/split.
- PPO training via Stable-Baselines3 (with optional `sb3-contrib` action masking).
- Evaluation pipeline with EV metrics and replay export.
- Pygame replay UI that animates deals and actions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Evaluate

```bash
python evaluate.py
```

## Betting smoke experiment (log-bankroll reward)

```bash
python train.py --device cpu --masking --episode-mode shoe --max-rounds-per-episode 200 --obs-version 4 --enable-betting --bet-levels 1,2,4,8 --bankroll-start 200 --betting-reward-mode log_bankroll --timesteps 2000000 --n-envs 8 --vec-env subproc --torch-threads 6 --torch-interop-threads 2 --train-eval-freq 500000 --train-eval-hands 20000 --model-in ./models/bj_obs4_bet_pretrain_strong.zip --model-out ./models/bj_obs4_bet_logbankroll_smoke_2m
```

```bash
python evaluate.py --model ./models/bj_obs4_bet_logbankroll_smoke_2m.zip --hands 200000 --algo auto --episode-mode shoe --max-rounds-per-episode 200 --obs-version 4 --enable-betting --bet-levels 1,2,4,8 --bankroll-start 200 --save-replays 0 --no-progress
```

## Replay UI

```bash
python replay_ui.py --replay replays/eval_bundle.json
```

## Files

- `blackjack_env.py` – environment and game rules.
- `utils/cards.py` – card and shoe logic.
- `train.py` – PPO training entrypoint.
- `evaluate.py` – policy evaluation and replay generation.
- `replay_logger.py` – replay JSON load/save helpers.
- `replay_ui.py` – animated replay viewer.
