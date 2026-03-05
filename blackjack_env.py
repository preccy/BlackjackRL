"""Gymnasium-compatible Blackjack environment with casino-style split/double rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.cards import Card, Shoe

ACTION_NAMES = {0: "stand", 1: "hit", 2: "double", 3: "split"}


@dataclass
class HandState:
    cards: List[Card] = field(default_factory=list)
    bet: float = 1.0
    done: bool = False
    is_split_aces: bool = False
    doubled: bool = False
    actions: List[str] = field(default_factory=list)


class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_decks: int = 6,
        penetration: float = 0.25,
        dealer_stands_soft17: bool = True,
        blackjack_payout: float = 1.5,
        das: bool = True,
        max_hands: int = 4,
        illegal_action_penalty: float = -0.05,
        seed: Optional[int] = None,
        record_events: bool = False,
    ):
        super().__init__()
        self.n_decks = n_decks
        self.penetration = penetration
        self.dealer_stands_soft17 = dealer_stands_soft17
        self.blackjack_payout = blackjack_payout
        self.das = das
        self.max_hands = max_hands
        self.illegal_action_penalty = illegal_action_penalty
        self.record_events = record_events

        self.rng = random.Random(seed)
        self.shoe = Shoe(n_decks=n_decks, penetration=penetration, rng=self.rng)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        self.hands: List[HandState] = []
        self.dealer_cards: List[Card] = []
        self.current_hand_idx = 0
        self.terminated = False
        self.round_step = 0
        self.events: List[dict] = []
        self.last_info: Dict[str, Any] = {}

    def _log(self, event_type: str, **payload: Any) -> None:
        if self.record_events:
            self.events.append({"step": self.round_step, "type": event_type, **payload})

    @staticmethod
    def hand_value(cards: List[Card]) -> tuple[int, bool]:
        total = sum(c.value for c in cards)
        soft_aces = sum(1 for c in cards if c.rank == "A")
        while total > 21 and soft_aces > 0:
            total -= 10
            soft_aces -= 1
        return total, soft_aces > 0


    @staticmethod
    def is_blackjack(cards: List[Card]) -> bool:
        return len(cards) == 2 and BlackjackEnv.hand_value(cards)[0] == 21

    def _dealer_upcard_value(self) -> int:
        if not self.dealer_cards:
            return 0
        return min(10, self.dealer_cards[0].value)

    def _can_double(self, hand: HandState) -> bool:
        if hand.done:
            return False
        return len(hand.cards) == 2 and (self.das or len(self.hands) == 1)

    def _can_split(self, hand: HandState) -> bool:
        if hand.done or len(hand.cards) != 2:
            return False
        if len(self.hands) >= self.max_hands:
            return False
        a, b = hand.cards
        if hand.is_split_aces:
            return False
        return a.rank_value == b.rank_value

    def _is_first_decision(self, hand: HandState) -> bool:
        return len(hand.actions) == 0

    def action_masks(self) -> np.ndarray:
        if self.terminated:
            return np.array([True, False, False, False], dtype=bool)
        hand = self.hands[self.current_hand_idx]
        return np.array(
            [
                True,
                not hand.done,
                self._can_double(hand),
                self._can_split(hand),
            ],
            dtype=bool,
        )

    def _obs(self) -> np.ndarray:
        hand = self.hands[self.current_hand_idx]
        total, usable = self.hand_value(hand.cards)
        total = min(total, 22)
        num_cards = min(len(hand.cards), 8)
        remaining = sum(1 for h in self.hands[self.current_hand_idx + 1 :] if not h.done)
        obs = np.array(
            [
                total / 22.0,
                float(usable),
                self._dealer_upcard_value() / 10.0,
                float(self._can_double(hand)),
                float(self._can_split(hand)),
                num_cards / 8.0,
                float(self._is_first_decision(hand)),
                remaining / 3.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _draw_to_hand(self, hand: HandState, owner: str, hand_index: int) -> None:
        card = self.shoe.draw()
        hand.cards.append(card)
        self._log("deal_card", to=owner, hand_index=hand_index, card=card.to_dict())

    def _deal_initial(self) -> None:
        self.hands = [HandState()]
        self.dealer_cards = []
        self.current_hand_idx = 0
        for _ in range(2):
            self._draw_to_hand(self.hands[0], "player", 0)
            d_card = self.shoe.draw()
            self.dealer_cards.append(d_card)
            self._log("deal_card", to="dealer", hand_index=0, card=d_card.to_dict())

    def _advance_hand(self) -> None:
        while self.current_hand_idx < len(self.hands) and self.hands[self.current_hand_idx].done:
            self.current_hand_idx += 1
            if self.current_hand_idx < len(self.hands):
                self._log("hand_transition", next_hand_index=self.current_hand_idx)

    def _dealer_should_hit(self) -> bool:
        total, usable = self.hand_value(self.dealer_cards)
        if total < 17:
            return True
        if total > 17:
            return False
        if total == 17 and usable and not self.dealer_stands_soft17:
            return True
        return False

    def _resolve_round(self) -> float:
        dealer_total, _ = self.hand_value(self.dealer_cards)
        while self._dealer_should_hit():
            c = self.shoe.draw()
            self.dealer_cards.append(c)
            self._log("dealer_play", action="hit", card=c.to_dict())
            dealer_total, _ = self.hand_value(self.dealer_cards)
        self._log("dealer_play", action="stand", dealer_total=dealer_total)

        total_reward = 0.0
        outcomes = []
        for idx, hand in enumerate(self.hands):
            player_total, _ = self.hand_value(hand.cards)
            p_bj = self.is_blackjack(hand.cards) and len(self.hands) == 1
            d_bj = self.is_blackjack(self.dealer_cards)
            if player_total > 21:
                reward, outcome = -hand.bet, "loss_bust"
            elif d_bj and p_bj:
                reward, outcome = 0.0, "push_blackjack"
            elif d_bj:
                reward, outcome = -hand.bet, "loss_dealer_blackjack"
            elif p_bj:
                reward, outcome = hand.bet * self.blackjack_payout, "win_blackjack"
            elif dealer_total > 21:
                reward, outcome = hand.bet, "win_dealer_bust"
            elif player_total > dealer_total:
                reward, outcome = hand.bet, "win"
            elif player_total < dealer_total:
                reward, outcome = -hand.bet, "loss"
            else:
                reward, outcome = 0.0, "push"
            total_reward += reward
            outcomes.append({"hand_index": idx, "outcome": outcome, "reward": reward, "bet": hand.bet})

        self._log("final_result", total_reward=total_reward, outcomes=outcomes)
        self.last_info = {
            "outcomes": outcomes,
            "total_reward": total_reward,
            "final_bankroll_delta": total_reward,
            "dealer_hand": [c.to_dict() for c in self.dealer_cards],
            "player_hands": [[c.to_dict() for c in hand.cards] for hand in self.hands],
            "actions_taken": [hand.actions[:] for hand in self.hands],
            "events": self.events[:],
            "shoe_meta": self.shoe.to_meta(),
        }
        return total_reward

    def _dealer_peek_blackjack(self) -> bool:
        if not self.dealer_cards:
            return False
        up = self.dealer_cards[0]
        if up.rank not in {"A", "T", "J", "Q", "K"}:
            return False
        return self.is_blackjack(self.dealer_cards)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.terminated = False
        self.round_step = 0
        self.events = []
        self._deal_initial()
        self._log(
            "initial_state",
            player_initial=[c.to_dict() for c in self.hands[0].cards],
            dealer_upcard=self.dealer_cards[0].to_dict(),
            dealer_hole_card=self.dealer_cards[1].to_dict(),
            shoe_meta=self.shoe.to_meta(),
        )

        if self._dealer_peek_blackjack() or self.is_blackjack(self.hands[0].cards):
            self.hands[0].done = True
            self.current_hand_idx = 1
            reward = self._resolve_round()
            self.terminated = True
            return np.zeros(self.observation_space.shape, dtype=np.float32), {**self.last_info, "immediate_reward": reward}

        return self._obs(), {"shoe_meta": self.shoe.to_meta(), "action_mask": self.action_masks()}

    def step(self, action: int):
        if self.terminated:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, self.last_info

        self.round_step += 1
        hand = self.hands[self.current_hand_idx]
        masks = self.action_masks()
        reward = 0.0
        if not masks[action]:
            reward += self.illegal_action_penalty
            action = 0

        hand.actions.append(ACTION_NAMES[action])
        self._log("action", hand_index=self.current_hand_idx, action_name=ACTION_NAMES[action])

        if action == 0:  # stand
            hand.done = True
        elif action == 1:  # hit
            self._draw_to_hand(hand, "player", self.current_hand_idx)
            total, _ = self.hand_value(hand.cards)
            if total >= 21:
                hand.done = True
        elif action == 2:  # double
            hand.bet *= 2
            hand.doubled = True
            self._draw_to_hand(hand, "player", self.current_hand_idx)
            hand.done = True
        elif action == 3:  # split
            c1, c2 = hand.cards
            left = HandState(cards=[c1], bet=hand.bet)
            right = HandState(cards=[c2], bet=hand.bet)
            split_aces = c1.rank == "A" and c2.rank == "A"
            left.is_split_aces = split_aces
            right.is_split_aces = split_aces
            self.hands[self.current_hand_idx] = left
            self.hands.insert(self.current_hand_idx + 1, right)
            self._draw_to_hand(left, "player", self.current_hand_idx)
            self._draw_to_hand(right, "player", self.current_hand_idx + 1)
            if split_aces:
                left.done = True
                right.done = True

        self._advance_hand()
        if self.current_hand_idx >= len(self.hands):
            reward += self._resolve_round()
            self.terminated = True
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, {
                **self.last_info,
                "action_mask": np.array([True, False, False, False], dtype=bool),
            }

        info = {"action_mask": self.action_masks()}
        return self._obs(), reward, False, False, info

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            return np.zeros((200, 300, 3), dtype=np.uint8)
        print("Dealer:", " ".join(str(c) for c in self.dealer_cards))
        print("Hands:")
        for i, hand in enumerate(self.hands):
            marker = "<-" if i == self.current_hand_idx and not self.terminated else ""
            total, _ = self.hand_value(hand.cards)
            print(i, [str(c) for c in hand.cards], f"total={total}", marker)

    def export_episode(self) -> dict:
        return {
            "shoe_meta": self.shoe.to_meta(),
            "dealer_hand": [c.to_dict() for c in self.dealer_cards],
            "player_hands": [[c.to_dict() for c in h.cards] for h in self.hands],
            "events": self.events[:],
            "info": self.last_info,
        }


def _assert_hand_value_examples() -> None:
    """Lightweight sanity checks for soft-hand handling."""
    mk = lambda ranks: [Card(rank=r, suit="♠") for r in ranks]
    assert BlackjackEnv.hand_value(mk(["A", "6"])) == (17, True)
    assert BlackjackEnv.hand_value(mk(["A", "9"])) == (20, True)
    assert BlackjackEnv.hand_value(mk(["A", "9", "5"])) == (15, False)
    assert BlackjackEnv.hand_value(mk(["A", "A", "9"])) == (21, True)
    assert BlackjackEnv.hand_value(mk(["A", "A", "9", "9"])) == (20, False)
