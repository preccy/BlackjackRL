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
