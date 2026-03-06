"""Gymnasium-compatible Blackjack environment with casino-style split/double rules."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
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
        obs_version: int = 1,
        episode_mode: str = "hand",
        max_rounds_per_episode: int = 200,
        shuffle_on_reset: bool = True,
        enable_betting: bool = False,
        bet_levels: Optional[List[float]] = None,
        bankroll_start: Optional[float] = None,
        bankroll_stop_on_zero: bool = False,
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
        if obs_version not in {1, 2, 3, 4}:
            raise ValueError(f"Unsupported obs_version={obs_version}; expected 1, 2, 3, or 4.")
        if episode_mode not in {"hand", "shoe"}:
            raise ValueError(f"Unsupported episode_mode={episode_mode}; expected 'hand' or 'shoe'.")
        self.obs_version = obs_version
        self.episode_mode = episode_mode
        self.max_rounds_per_episode = max(1, int(max_rounds_per_episode))
        self.shuffle_on_reset = bool(shuffle_on_reset)
        self.enable_betting = bool(enable_betting)
        raw_bet_levels = bet_levels if bet_levels is not None else [1.0]
        self.bet_levels = [float(level) for level in raw_bet_levels]
        if not self.bet_levels or any(level <= 0 for level in self.bet_levels):
            raise ValueError("bet_levels must contain one or more positive values")
        self.current_bet = self.bet_levels[0]
        self.phase = "BET" if self.enable_betting else "PLAY"
        self.bankroll_start = bankroll_start
        self.bankroll_current = float(bankroll_start) if bankroll_start is not None else 0.0
        self.bankroll_stop_on_zero = bool(bankroll_stop_on_zero)

        self.rng = random.Random(seed)
        self.shoe = Shoe(n_decks=n_decks, penetration=penetration, rng=self.rng)

        total_actions = 4 + len(self.bet_levels) if self.enable_betting else 4
        self.action_space = spaces.Discrete(total_actions)
        if self.obs_version == 1:
            obs_size = 8
        elif self.obs_version in {2, 3}:
            obs_size = 9 if self.obs_version == 2 else 20
        else:
            obs_size = 22
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.hands: List[HandState] = []
        self.dealer_cards: List[Card] = []
        self.current_hand_idx = 0
        self.terminated = False
        self.round_step = 0
        self.events: List[dict] = []
        self.last_info: Dict[str, Any] = {}
        self._pending_terminal_reward: Optional[float] = None
        self.rounds_in_episode = 0
        self.last_round_revealed_cards: List[Card] = []
        self._reshuffle_happened_this_round = False
        self.last_shuffle_count = self.shoe.shuffle_count
        self.rounds_since_shuffle = 0
        self.cum_revealed_bins = np.zeros(10, dtype=np.float32)

    def _reset_cum_bins_on_shuffle(self) -> None:
        self.cum_revealed_bins = np.zeros(10, dtype=np.float32)
        self.rounds_since_shuffle = 0

    @staticmethod
    def _rank_to_bin(rank: str) -> int:
        order = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8}
        return order.get(rank, 9)

    def _update_cum_bins(self, cards: List[Card]) -> None:
        for card in cards:
            self.cum_revealed_bins[self._rank_to_bin(card.rank)] += 1.0

    def _terminated_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        return mask

    def _cards_remaining_norm(self) -> float:
        deck_size = max(1, 52 * self.n_decks)
        return min(1.0, self.shoe.cards_remaining / deck_size)

    def _cum_rank_bins_norm(self) -> List[float]:
        deck_size = max(1, 52 * self.n_decks)
        return [min(1.0, float(v) / deck_size) for v in self.cum_revealed_bins.tolist()]

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
            return self._terminated_action_mask()
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.enable_betting and self.phase == "BET":
            mask[4 : 4 + len(self.bet_levels)] = True
            return mask
        hand = self.hands[self.current_hand_idx]
        mask[0] = True
        mask[1] = not hand.done
        mask[2] = self._can_double(hand)
        mask[3] = self._can_split(hand)
        return mask

    def _obs(self) -> np.ndarray:
        if self.enable_betting and self.phase == "BET":
            dummy = Card(rank="2", suit="♠")
            return self.obs_from_cards(
                player_cards=[],
                dealer_upcard=dummy,
                force_can_split=False,
                force_can_double=False,
                remaining_hands=0,
                is_first_decision=True,
                phase_is_bet=True,
            )
        if not self.hands or self.current_hand_idx >= len(self.hands):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        hand = self.hands[self.current_hand_idx]
        return self.obs_from_cards(
            player_cards=hand.cards,
            dealer_upcard=self.dealer_cards[0],
            force_can_split=self._can_split(hand),
            force_can_double=self._can_double(hand),
            remaining_hands=sum(1 for h in self.hands[self.current_hand_idx + 1 :] if not h.done),
            is_first_decision=self._is_first_decision(hand),
            phase_is_bet=False,
        )

    def obs_from_cards(
        self,
        player_cards: List[Card],
        dealer_upcard: Card,
        force_can_split: Optional[bool] = None,
        force_can_double: Optional[bool] = None,
        remaining_hands: int = 0,
        is_first_decision: bool = True,
        phase_is_bet: bool = False,
    ) -> np.ndarray:
        total, usable = self.hand_value(player_cards)
        if phase_is_bet:
            total = 0
            usable = False
        total = min(total, 22)
        num_cards = min(len(player_cards), 8)
        can_double = bool(force_can_double) if force_can_double is not None else (len(player_cards) == 2)
        can_split = False
        if force_can_split is not None:
            can_split = bool(force_can_split)
        elif len(player_cards) == 2:
            can_split = player_cards[0].rank_value == player_cards[1].rank_value

        obs_vals = [
            total / 22.0,
            float(usable),
            (0.0 if phase_is_bet else min(10, dealer_upcard.value) / 10.0),
            float(can_double),
            float(can_split),
            num_cards / 8.0,
            float(is_first_decision),
            min(remaining_hands, 3) / 3.0,
        ]
        if self.obs_version in {2, 3, 4}:
            obs_vals.append((0.0 if phase_is_bet else float(dealer_upcard.rank == "A")))
        if self.obs_version == 3:
            obs_vals.extend(self._last_round_rank_bins())
            obs_vals.append(self._cards_remaining_norm())
        elif self.obs_version == 4:
            obs_vals.extend(self._cum_rank_bins_norm())
            obs_vals.append(self._cards_remaining_norm())
            obs_vals.append(min(1.0, self.rounds_since_shuffle / max(1, self.max_rounds_per_episode)))
            obs_vals.append(float(phase_is_bet))
        obs = np.array(obs_vals, dtype=np.float32)
        return obs

    def _last_round_rank_bins(self) -> List[float]:
        bins = {"A": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "T": 0}
        for card in self.last_round_revealed_cards:
            rank = card.rank
            if rank in {"T", "J", "Q", "K"}:
                bins["T"] += 1
            elif rank in bins:
                bins[rank] += 1
        return [min(20, bins[key]) / 20.0 for key in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T"]]

    def _draw_card(self) -> Card:
        prev_shuffle_count = self.shoe.shuffle_count
        card = self.shoe.draw()
        if self.shoe.shuffle_count != prev_shuffle_count:
            self._reshuffle_happened_this_round = True
            self.last_shuffle_count = self.shoe.shuffle_count
            self._reset_cum_bins_on_shuffle()
        return card

    def _draw_to_hand(self, hand: HandState, owner: str, hand_index: int) -> None:
        card = self._draw_card()
        hand.cards.append(card)
        self._log("deal_card", to=owner, hand_index=hand_index, card=card.to_dict())

    def _deal_initial(self) -> None:
        self._reshuffle_happened_this_round = False
        self.hands = [HandState(bet=self.current_bet)]
        self.dealer_cards = []
        self.current_hand_idx = 0
        for _ in range(2):
            self._draw_to_hand(self.hands[0], "player", 0)
            d_card = self._draw_card()
            self.dealer_cards.append(d_card)
            self._log("deal_card", to="dealer", hand_index=0, card=d_card.to_dict())

    def _collect_revealed_cards(self) -> List[Card]:
        cards = self.dealer_cards[:]
        for hand in self.hands:
            cards.extend(hand.cards)
        return cards

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
            c = self._draw_card()
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
            "total_wagered": float(sum(hand.bet for hand in self.hands)),
            "current_bet": float(self.current_bet),
            "final_bankroll_delta": total_reward,
            "dealer_hand": [c.to_dict() for c in self.dealer_cards],
            "player_hands": [[c.to_dict() for c in hand.cards] for hand in self.hands],
            "actions_taken": [hand.actions[:] for hand in self.hands],
            "events": self.events[:],
            "shoe_meta": self.shoe.to_meta(),
        }
        return total_reward

    def _build_round_end_payload(self, round_reward: float, reshuffle_happened: bool) -> Dict[str, Any]:
        last_info = copy.deepcopy(self.last_info)
        return {
            "outcomes": last_info.get("outcomes", []),
            "total_reward": last_info.get("total_reward", round_reward),
            "dealer_hand": last_info.get("dealer_hand", []),
            "player_hands": last_info.get("player_hands", []),
            "actions_taken": last_info.get("actions_taken", []),
            "events": last_info.get("events", []),
            "shoe_meta": last_info.get("shoe_meta", self.shoe.to_meta()),
            "round_end": True,
            "round_reward": round_reward,
            "total_wagered": last_info.get("total_wagered", 0.0),
            "current_bet": last_info.get("current_bet", float(self.current_bet)),
            "rounds_in_episode": self.rounds_in_episode,
            "reshuffle_happened": reshuffle_happened,
            "round_replay": {
                "shoe_meta": last_info.get("shoe_meta", self.shoe.to_meta()),
                "dealer_hand": last_info.get("dealer_hand", []),
                "player_hands": last_info.get("player_hands", []),
                "events": last_info.get("events", []),
                "info": last_info,
            },
        }

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
        self._pending_terminal_reward = None
        self.round_step = 0
        self.rounds_in_episode = 0
        self.last_round_revealed_cards = []
        self.events = []
        self.current_bet = self.bet_levels[0]
        self.phase = "BET" if self.enable_betting else "PLAY"
        if self.bankroll_start is not None:
            self.bankroll_current = float(self.bankroll_start)

        if self.episode_mode == "shoe" and self.shuffle_on_reset:
            self.shoe._init_cards()

        if self.shoe.shuffle_count != self.last_shuffle_count:
            self.last_shuffle_count = self.shoe.shuffle_count
            self._reset_cum_bins_on_shuffle()

        obs, info, auto_reward, terminated = self._start_round()
        if terminated:
            self.terminated = True
            self._pending_terminal_reward = auto_reward
            return np.zeros(self.observation_space.shape, dtype=np.float32), {**info, "immediate_reward": auto_reward}
        if auto_reward != 0.0:
            info = {**info, "immediate_reward": auto_reward}
        return obs, info

    def _finalize_round(self, round_reward: float, reshuffle_happened: bool) -> Dict[str, Any]:
        self.rounds_in_episode += 1
        self.last_round_revealed_cards = self._collect_revealed_cards()
        self._update_cum_bins(self.last_round_revealed_cards)
        self.rounds_since_shuffle += 1
        if self.bankroll_start is not None:
            self.bankroll_current += round_reward
        payload = self._build_round_end_payload(round_reward, reshuffle_happened)
        payload["total_wagered"] = float(sum(hand.bet for hand in self.hands))
        if self.bankroll_start is not None:
            payload["bankroll_current"] = float(self.bankroll_current)
        return payload

    def _start_round(self) -> tuple[np.ndarray, Dict[str, Any], float, bool]:
        if self.enable_betting:
            self.phase = "BET"
            self.hands = []
            self.dealer_cards = []
            info = {
                "shoe_meta": self.shoe.to_meta(),
                "action_mask": self.action_masks(),
                "rounds_in_episode": self.rounds_in_episode,
                "reshuffle_happened": False,
                "phase": self.phase,
            }
            return self._obs(), info, 0.0, False

        auto_reward = 0.0
        reshuffle_happened = False

        while True:
            self._deal_initial()
            reshuffle_happened = reshuffle_happened or self._reshuffle_happened_this_round
            self._log(
                "initial_state",
                player_initial=[c.to_dict() for c in self.hands[0].cards],
                dealer_upcard=self.dealer_cards[0].to_dict(),
                dealer_hole_card=self.dealer_cards[1].to_dict(),
                shoe_meta=self.shoe.to_meta(),
            )

            if self.episode_mode == "shoe" and reshuffle_happened and self.rounds_in_episode > 0:
                info = {
                    "shoe_meta": self.shoe.to_meta(),
                    "action_mask": self._terminated_action_mask(),
                    "reshuffle_happened": True,
                    "rounds_in_episode": self.rounds_in_episode,
                }
                return np.zeros(self.observation_space.shape, dtype=np.float32), info, auto_reward, True

            if self._dealer_peek_blackjack() or self.is_blackjack(self.hands[0].cards):
                self.hands[0].done = True
                self.current_hand_idx = 1
                round_reward = self._resolve_round()
                auto_reward += round_reward
                round_reshuffle = self._reshuffle_happened_this_round
                reshuffle_happened = reshuffle_happened or round_reshuffle
                round_payload = self._finalize_round(round_reward, round_reshuffle)
                max_rounds_hit = self.episode_mode == "shoe" and self.rounds_in_episode >= self.max_rounds_per_episode
                bankroll_bust = self.bankroll_start is not None and self.bankroll_stop_on_zero and self.bankroll_current <= 0
                if self.episode_mode == "hand" or round_reshuffle or max_rounds_hit or bankroll_bust:
                    info = {**round_payload, "action_mask": self._terminated_action_mask()}
                    return np.zeros(self.observation_space.shape, dtype=np.float32), info, auto_reward, True

                next_obs, next_info, next_auto_reward, next_terminated = self._start_round()
                auto_reward += next_auto_reward
                if next_terminated:
                    info = {
                        **next_info,
                        **round_payload,
                        "reshuffle_happened": bool(
                            round_payload["reshuffle_happened"] or next_info.get("reshuffle_happened", False)
                        ),
                        "action_mask": self._terminated_action_mask(),
                    }
                    return np.zeros(self.observation_space.shape, dtype=np.float32), info, auto_reward, True

                info = {
                    **next_info,
                    **round_payload,
                    "reshuffle_happened": bool(
                        round_payload["reshuffle_happened"] or next_info.get("reshuffle_happened", False)
                    ),
                }
                return next_obs, info, auto_reward, False

            self.phase = "PLAY"
            info = {
                "shoe_meta": self.shoe.to_meta(),
                "action_mask": self.action_masks(),
                "rounds_in_episode": self.rounds_in_episode,
                "reshuffle_happened": reshuffle_happened,
                "phase": self.phase,
            }
            return self._obs(), info, auto_reward, False

    def step(self, action: int):
        if self.terminated:
            reward = 0.0
            if self._pending_terminal_reward is not None:
                reward = self._pending_terminal_reward
                self._pending_terminal_reward = None
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, self.last_info

        self.round_step += 1
        masks = self.action_masks()
        reward = 0.0
        if action < 0 or action >= self.action_space.n or not masks[action]:
            reward += self.illegal_action_penalty
            action = int(np.where(masks)[0][0]) if np.any(masks) else 0

        if self.enable_betting and self.phase == "BET":
            bet_idx = max(0, action - 4)
            self.current_bet = self.bet_levels[bet_idx]
            self.phase = "PLAY"
            self._deal_initial()
            self._log(
                "initial_state",
                player_initial=[c.to_dict() for c in self.hands[0].cards],
                dealer_upcard=self.dealer_cards[0].to_dict(),
                dealer_hole_card=self.dealer_cards[1].to_dict(),
                shoe_meta=self.shoe.to_meta(),
                selected_bet=self.current_bet,
            )

            if self._dealer_peek_blackjack() or self.is_blackjack(self.hands[0].cards):
                self.hands[0].done = True
                self.current_hand_idx = 1
                round_reward = self._resolve_round()
                reward += round_reward
                round_reshuffle = self._reshuffle_happened_this_round
                round_payload = self._finalize_round(round_reward, round_reshuffle)
                max_rounds_hit = self.episode_mode == "shoe" and self.rounds_in_episode >= self.max_rounds_per_episode
                bankroll_bust = self.bankroll_start is not None and self.bankroll_stop_on_zero and self.bankroll_current <= 0

                if self.episode_mode == "hand" or round_reshuffle or max_rounds_hit or bankroll_bust:
                    self.terminated = True
                    return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, {
                        **round_payload,
                        "action_mask": self._terminated_action_mask(),
                        "phase": "TERMINATED",
                    }

                next_obs, next_info, next_auto_reward, next_terminated = self._start_round()
                reward += next_auto_reward
                if next_terminated:
                    self.terminated = True
                    return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, {
                        **next_info,
                        **round_payload,
                        "reshuffle_happened": bool(
                            round_payload["reshuffle_happened"] or next_info.get("reshuffle_happened", False)
                        ),
                        "action_mask": self._terminated_action_mask(),
                        "phase": "TERMINATED",
                    }

                info = {
                    **next_info,
                    **round_payload,
                    "reshuffle_happened": bool(
                        round_payload["reshuffle_happened"] or next_info.get("reshuffle_happened", False)
                    ),
                }
                return next_obs, reward, False, False, info

            info = {
                "action_mask": self.action_masks(),
                "shoe_meta": self.shoe.to_meta(),
                "rounds_in_episode": self.rounds_in_episode,
                "reshuffle_happened": self._reshuffle_happened_this_round,
                "phase": self.phase,
                "current_bet": self.current_bet,
            }
            return self._obs(), reward, False, False, info

        hand = self.hands[self.current_hand_idx]
        hand.actions.append(ACTION_NAMES[action])
        self._log("action", hand_index=self.current_hand_idx, action_name=ACTION_NAMES[action])

        if action == 0:
            hand.done = True
        elif action == 1:
            self._draw_to_hand(hand, "player", self.current_hand_idx)
            total, _ = self.hand_value(hand.cards)
            if total >= 21:
                hand.done = True
        elif action == 2:
            hand.bet *= 2
            hand.doubled = True
            self._draw_to_hand(hand, "player", self.current_hand_idx)
            hand.done = True
        elif action == 3:
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
            round_reward = self._resolve_round()
            reward += round_reward
            reshuffle_happened = self._reshuffle_happened_this_round
            round_payload = self._finalize_round(round_reward, reshuffle_happened)
            max_rounds_hit = self.episode_mode == "shoe" and self.rounds_in_episode >= self.max_rounds_per_episode
            bankroll_bust = self.bankroll_start is not None and self.bankroll_stop_on_zero and self.bankroll_current <= 0

            if self.episode_mode == "hand" or reshuffle_happened or max_rounds_hit or bankroll_bust:
                self.terminated = True
                return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, {
                    **round_payload,
                    "action_mask": self._terminated_action_mask(),
                }

            next_obs, next_info, auto_reward, auto_terminated = self._start_round()
            reward += auto_reward
            if auto_terminated:
                self.terminated = True
                return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, {
                    **round_payload,
                    **next_info,
                    "reshuffle_happened": bool(round_payload["reshuffle_happened"] or next_info.get("reshuffle_happened", False)),
                    "action_mask": self._terminated_action_mask(),
                }

            return next_obs, reward, False, False, {
                **round_payload,
                **next_info,
                "reshuffle_happened": bool(round_payload["reshuffle_happened"] or next_info.get("reshuffle_happened", False)),
            }

        info = {
            "action_mask": self.action_masks(),
            "shoe_meta": self.shoe.to_meta(),
            "rounds_in_episode": self.rounds_in_episode,
            "reshuffle_happened": self._reshuffle_happened_this_round,
            "phase": self.phase,
            "current_bet": self.current_bet,
        }
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
