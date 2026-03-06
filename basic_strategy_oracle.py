from __future__ import annotations

from typing import Iterable

# Actions: 0=STAND, 1=HIT, 2=DOUBLE, 3=SPLIT
STAND, HIT, DOUBLE, SPLIT = 0, 1, 2, 3


def _dealer_key(dealer_upcard_rank: str) -> str:
    rank = dealer_upcard_rank.upper()
    if rank in {"J", "Q", "K"}:
        return "T"
    if rank in {"2", "3", "4", "5", "6", "7", "8", "9", "T", "A"}:
        return rank
    raise ValueError(f"Unsupported dealer upcard rank: {dealer_upcard_rank}")


def _is_pair(player_cards: Iterable) -> bool:
    cards = list(player_cards)
    if len(cards) != 2:
        return False
    return cards[0].rank_value == cards[1].rank_value


def _pair_rank(player_cards: Iterable) -> str:
    c0 = list(player_cards)[0]
    return "T" if c0.rank in {"T", "J", "Q", "K"} else c0.rank


def _hand_value(player_cards: Iterable) -> tuple[int, bool]:
    cards = list(player_cards)
    total = sum(c.value for c in cards)
    soft_aces = sum(1 for c in cards if c.rank == "A")
    while total > 21 and soft_aces > 0:
        total -= 10
        soft_aces -= 1
    return total, soft_aces > 0


def _pair_action(pair: str, dealer: str) -> int:
    if pair == "A":
        return SPLIT
    if pair == "T":
        return STAND
    if pair == "9":
        return SPLIT if dealer in {"2", "3", "4", "5", "6", "8", "9"} else STAND
    if pair == "8":
        return SPLIT
    if pair == "7":
        return SPLIT if dealer in {"2", "3", "4", "5", "6", "7"} else HIT
    if pair == "6":
        return SPLIT if dealer in {"2", "3", "4", "5", "6"} else HIT
    if pair == "5":
        return DOUBLE if dealer in {"2", "3", "4", "5", "6", "7", "8", "9"} else HIT
    if pair == "4":
        return SPLIT if dealer in {"5", "6"} else HIT
    if pair in {"2", "3"}:
        return SPLIT if dealer in {"2", "3", "4", "5", "6", "7"} else HIT
    return HIT


def _soft_action(total: int, dealer: str) -> int:
    if total >= 20:
        return STAND
    if total == 19:
        return DOUBLE if dealer == "6" else STAND
    if total == 18:
        if dealer in {"3", "4", "5", "6"}:
            return DOUBLE
        if dealer in {"2", "7", "8"}:
            return STAND
        return HIT
    if total == 17:
        return DOUBLE if dealer in {"3", "4", "5", "6"} else HIT
    if total in {15, 16}:
        return DOUBLE if dealer in {"4", "5", "6"} else HIT
    if total in {13, 14}:
        return DOUBLE if dealer in {"5", "6"} else HIT
    return HIT


def _hard_action(total: int, dealer: str) -> int:
    if total >= 17:
        return STAND
    if total in {13, 14, 15, 16}:
        return STAND if dealer in {"2", "3", "4", "5", "6"} else HIT
    if total == 12:
        return STAND if dealer in {"4", "5", "6"} else HIT
    if total == 11:
        return DOUBLE
    if total == 10:
        return DOUBLE if dealer in {"2", "3", "4", "5", "6", "7", "8", "9"} else HIT
    if total == 9:
        return DOUBLE if dealer in {"3", "4", "5", "6"} else HIT
    return HIT


def oracle_action(player_cards, dealer_upcard_rank: str, can_double: bool, can_split: bool) -> int:
    """Return basic-strategy action for multi-deck S17, DAS, no surrender."""
    cards = list(player_cards)
    dealer = _dealer_key(dealer_upcard_rank)

    action = None
    if _is_pair(cards):
        action = _pair_action(_pair_rank(cards), dealer)

    if action is None:
        total, soft = _hand_value(cards)
        if soft and len(cards) == 2 and total <= 20:
            action = _soft_action(total, dealer)
        else:
            action = _hard_action(total, dealer)

    if action == DOUBLE and not can_double:
        return HIT
    if action == SPLIT and not can_split:
        return HIT
    return action
