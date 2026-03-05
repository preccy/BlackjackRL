"""Card and shoe utilities for blackjack."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
SUITS = ["♠", "♥", "♦", "♣"]


@dataclass(frozen=True)
class Card:
    """Simple playing card with blackjack helpers."""

    rank: str
    suit: str

    @property
    def value(self) -> int:
        if self.rank == "A":
            return 11
        if self.rank in {"T", "J", "Q", "K"}:
            return 10
        return int(self.rank)

    @property
    def rank_value(self) -> int:
        """Value used for pair matching in split rules."""
        if self.rank in {"T", "J", "Q", "K"}:
            return 10
        if self.rank == "A":
            return 1
        return int(self.rank)

    def to_dict(self) -> dict:
        return {"rank": self.rank, "suit": self.suit, "value": self.value}

    @staticmethod
    def from_dict(data: dict) -> "Card":
        return Card(rank=data["rank"], suit=data["suit"])

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


class Shoe:
    """Multi-deck blackjack shoe with penetration-based reshuffle."""

    def __init__(self, n_decks: int = 6, penetration: float = 0.25, rng: random.Random | None = None):
        self.n_decks = n_decks
        self.penetration = penetration
        self.rng = rng or random.Random()
        self.cards: List[Card] = []
        self.shuffle_count = 0
        self._init_cards()

    def _init_cards(self) -> None:
        self.cards = [Card(rank=r, suit=s) for _ in range(self.n_decks) for s in SUITS for r in RANKS]
        self.rng.shuffle(self.cards)
        self.shuffle_count += 1

    @property
    def cards_remaining(self) -> int:
        return len(self.cards)

    @property
    def reshuffle_cutoff(self) -> int:
        return int(52 * self.n_decks * self.penetration)

    def needs_reshuffle(self) -> bool:
        return self.cards_remaining < self.reshuffle_cutoff

    def draw(self) -> Card:
        if self.needs_reshuffle() or not self.cards:
            self._init_cards()
        return self.cards.pop()

    def to_meta(self) -> dict:
        return {
            "n_decks": self.n_decks,
            "penetration": self.penetration,
            "cards_remaining": self.cards_remaining,
            "reshuffle_cutoff": self.reshuffle_cutoff,
            "shuffle_count": self.shuffle_count,
        }
