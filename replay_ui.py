from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pygame

from replay_logger import load_replay

WIDTH, HEIGHT = 1280, 800
CARD_W, CARD_H = 96, 136
TABLE_GREEN = (17, 73, 45)
TABLE_GREEN_DARK = (10, 48, 29)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GOLD = (229, 190, 89)
GRAY = (145, 152, 158)
RED = (214, 58, 66)
BLUE = (70, 130, 225)
YELLOW = (244, 208, 63)
PURPLE = (150, 92, 220)
PANEL_BG = (18, 25, 31)
ACTION_COLORS = {
    "Hit": YELLOW,
    "Stand": WHITE,
    "Double": BLUE,
    "Split": PURPLE,
}
BUTTONS = ["Hit", "Stand", "Double", "Split"]
STATE_DEAL = "DEAL"
STATE_PLAYER_ACTION = "PLAYER_ACTION"
STATE_DEALER_PLAY = "DEALER_PLAY"
STATE_RESULT = "RESULT"


class ReplayUI:
    def __init__(self, episodes: list[dict], speed: float = 1.0):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Blackjack Replay")
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("georgia", 40, bold=True)
        self.action_font = pygame.font.SysFont("arial", 42, bold=True)
        self.label_font = pygame.font.SysFont("arial", 28, bold=True)
        self.stat_font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.card_rank_font = pygame.font.SysFont("arial", 30, bold=True)
        self.card_suit_font = pygame.font.SysFont("arial", 42)

        self.episodes = episodes
        self.episode_idx = 0
        self.speed = speed
        self.delay_ms = 320
        self.last_tick = 0
        self.playing = True

        self.reset_episode()

    def reset_episode(self):
        episode = self.episodes[self.episode_idx]
        self.events = episode.get("events", [])
        self.event_ptr = 0

        self.dealer: list[dict] = []
        self.hands: dict[int, list[dict]] = {0: []}
        self.bets: dict[int, float] = {0: 1.0}
        self.outcomes: list[dict] = []
        self.total_reward = 0.0
        self.active_hand = 0
        self.highlight_action: str | None = None
        self.dealer_trace: list[str] = []
        self.play_state = STATE_DEAL
        self.result_overlay = ""

        self.animation: dict | None = None
        self.history: list[dict] = [self._snapshot()]

    def _snapshot(self) -> dict:
        return {
            "event_ptr": self.event_ptr,
            "dealer": copy.deepcopy(self.dealer),
            "hands": copy.deepcopy(self.hands),
            "bets": copy.deepcopy(self.bets),
            "outcomes": copy.deepcopy(self.outcomes),
            "total_reward": self.total_reward,
            "active_hand": self.active_hand,
            "highlight_action": self.highlight_action,
            "dealer_trace": self.dealer_trace[:],
            "play_state": self.play_state,
            "result_overlay": self.result_overlay,
        }

    def _restore(self, snap: dict):
        self.event_ptr = snap["event_ptr"]
        self.dealer = copy.deepcopy(snap["dealer"])
        self.hands = copy.deepcopy(snap["hands"])
        self.bets = copy.deepcopy(snap["bets"])
        self.outcomes = copy.deepcopy(snap["outcomes"])
        self.total_reward = snap["total_reward"]
        self.active_hand = snap["active_hand"]
        self.highlight_action = snap["highlight_action"]
        self.dealer_trace = snap["dealer_trace"][:]
        self.play_state = snap["play_state"]
        self.result_overlay = snap["result_overlay"]
        self.animation = None

    @staticmethod
    def hand_value(cards: list[dict]) -> tuple[int, bool]:
        total = 0
        aces = 0
        for c in cards:
            r = c["rank"]
            if r == "A":
                aces += 1
                total += 11
            elif r in {"T", "J", "Q", "K"}:
                total += 10
            else:
                total += int(r)
        usable_ace = False
        while total > 21 and aces:
            total -= 10
            aces -= 1
        if aces > 0:
            usable_ace = True
        return total, usable_ace

    def total_text(self, cards: list[dict], label: str) -> str:
        if not cards:
            return f"{label}: --"
        total, soft = self.hand_value(cards)
        ranks = ",".join(c["rank"] for c in cards)
        if soft:
            return f"{label}: {ranks} = Soft {total}"
        return f"{label}: {total}"

    def card_surface(self, card: dict) -> pygame.Surface:
        suit = card["suit"]
        color = RED if suit in ["♥", "♦"] else BLACK
        surf = pygame.Surface((CARD_W + 8, CARD_H + 8), pygame.SRCALPHA)
        pygame.draw.rect(surf, (0, 0, 0, 80), (4, 4, CARD_W, CARD_H), border_radius=13)
        pygame.draw.rect(surf, WHITE, (0, 0, CARD_W, CARD_H), border_radius=13)
        pygame.draw.rect(surf, BLACK, (0, 0, CARD_W, CARD_H), width=2, border_radius=13)

        rank = card["rank"]
        corner = self.card_rank_font.render(rank, True, color)
        suit_small = self.card_rank_font.render(suit, True, color)
        center = self.card_suit_font.render(suit, True, color)

        surf.blit(corner, (10, 8))
        surf.blit(suit_small, (12, 36))
        surf.blit(center, (CARD_W // 2 - center.get_width() // 2, CARD_H // 2 - center.get_height() // 2))

        corner_b = pygame.transform.flip(corner, True, True)
        suit_b = pygame.transform.flip(suit_small, True, True)
        surf.blit(corner_b, (CARD_W - corner_b.get_width() - 10, CARD_H - 44))
        surf.blit(suit_b, (CARD_W - suit_b.get_width() - 12, CARD_H - 72))
        return surf

    def hand_anchor(self, area_y: int, hand_pos: int, total_hands: int) -> tuple[int, int]:
        spacing = 290
        start_x = WIDTH // 2 - ((total_hands - 1) * spacing) // 2
        return start_x + hand_pos * spacing, area_y

    def start_card_animation(self, card: dict, owner: str, hand_index: int):
        if owner == "dealer":
            target_count = len(self.dealer)
            end_x = WIDTH // 2 - 140 + target_count * (CARD_W - 10)
            end_y = 170
        else:
            total_hands = max(1, len(self.hands))
            sorted_hands = sorted(self.hands)
            hand_pos = sorted_hands.index(hand_index)
            hx, hy = self.hand_anchor(520, hand_pos, total_hands)
            target_count = len(self.hands.get(hand_index, []))
            end_x = hx + target_count * (CARD_W - 14)
            end_y = hy

        self.animation = {
            "card": card,
            "owner": owner,
            "hand_index": hand_index,
            "start": (WIDTH - 130, 88),
            "end": (end_x, end_y),
            "start_time": pygame.time.get_ticks(),
            "duration": 250,
        }

    def apply_event(self, evt: dict):
        t = evt.get("type")
        if t == "deal_card":
            owner = evt["to"]
            hand_index = evt.get("hand_index", 0)
            if owner == "dealer":
                self.dealer.append(evt["card"])
                self.play_state = STATE_DEAL if self.play_state == STATE_DEAL else self.play_state
            else:
                self.hands.setdefault(hand_index, [])
                self.hands[hand_index].append(evt["card"])
        elif t == "action":
            self.active_hand = evt.get("hand_index", 0)
            self.highlight_action = evt.get("action_name", "").capitalize()
            self.play_state = STATE_PLAYER_ACTION
        elif t == "hand_transition":
            self.active_hand = evt.get("next_hand_index", 0)
        elif t == "dealer_play":
            self.play_state = STATE_DEALER_PLAY
            action = evt.get("action", "")
            if action == "hit" and "card" in evt:
                card = evt["card"]
                self.dealer.append(card)
                total, _ = self.hand_value(self.dealer)
                self.dealer_trace.append(f"Dealer hits, draws {card['rank']}{card['suit']} → {total}")
            elif action == "stand":
                total = evt.get("dealer_total", self.hand_value(self.dealer)[0])
                self.dealer_trace.append(f"Dealer stands on {total}")
        elif t == "final_result":
            self.outcomes = evt.get("outcomes", [])
            self.total_reward = evt.get("total_reward", 0.0)
            self.play_state = STATE_RESULT
            for out in self.outcomes:
                self.bets[out["hand_index"]] = out.get("bet", 1.0)
            self.result_overlay = self.result_text()

    def step_forward(self):
        if self.animation or self.event_ptr >= len(self.events):
            return

        evt = self.events[self.event_ptr]
        self.event_ptr += 1
        if evt.get("type") == "deal_card":
            self.start_card_animation(evt["card"], evt["to"], evt.get("hand_index", 0))
        else:
            self.apply_event(evt)
            self.history.append(self._snapshot())

    def step_back(self):
        if self.animation:
            self.animation = None
        if len(self.history) <= 1:
            return
        self.history.pop()
        self._restore(self.history[-1])

    def update(self):
        now = pygame.time.get_ticks()

        if self.animation:
            elapsed = now - self.animation["start_time"]
            if elapsed >= self.animation["duration"]:
                evt = self.events[self.event_ptr - 1]
                self.apply_event(evt)
                self.history.append(self._snapshot())
                self.animation = None

        if self.playing and not self.animation and now - self.last_tick > self.delay_ms / self.speed:
            self.step_forward()
            self.last_tick = now

    def result_text(self) -> str:
        if not self.outcomes:
            return ""
        names = {o["outcome"] for o in self.outcomes}
        if any("blackjack" in n and "win" in n for n in names):
            return "BLACKJACK!"
        if all(n.startswith("push") for n in names):
            return "PUSH"
        if self.total_reward > 0:
            return "PLAYER WINS"
        if self.total_reward < 0:
            return "DEALER WINS"
        return "PUSH"

    def draw_table(self):
        self.screen.fill(TABLE_GREEN_DARK)
        pygame.draw.ellipse(self.screen, TABLE_GREEN, (70, 70, WIDTH - 140, HEIGHT - 140))
        pygame.draw.arc(self.screen, GOLD, (140, 300, WIDTH - 280, 410), 3.6, 5.8, 4)
        title = self.title_font.render("BLACKJACK REPLAY", True, GOLD)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 18))

        shoe = pygame.Rect(WIDTH - 150, 80, 95, 140)
        pygame.draw.rect(self.screen, (46, 41, 36), shoe, border_radius=10)
        pygame.draw.rect(self.screen, GOLD, shoe, width=2, border_radius=10)
        self.screen.blit(self.small_font.render("SHOE", True, WHITE), (shoe.x + 24, shoe.y + 56))

    def draw_hand(self, cards: list[dict], x: int, y: int):
        for i, card in enumerate(cards):
            self.screen.blit(self.card_surface(card), (x + i * (CARD_W - 14), y))

    def draw(self):
        self.draw_table()

        dealer_lbl = self.label_font.render("Dealer", True, GOLD)
        self.screen.blit(dealer_lbl, (110, 120))
        self.draw_hand(self.dealer, WIDTH // 2 - 140, 170)
        dealer_total = self.stat_font.render(self.total_text(self.dealer, "Dealer"), True, WHITE)
        self.screen.blit(dealer_total, (WIDTH // 2 - dealer_total.get_width() // 2, 320))

        player_lbl = self.label_font.render("Player", True, GOLD)
        self.screen.blit(player_lbl, (110, 470))

        sorted_hands = sorted(self.hands)
        total_hands = max(1, len(sorted_hands))
        total_wagered = 0.0
        for pos, idx in enumerate(sorted_hands):
            hx, hy = self.hand_anchor(520, pos, total_hands)
            if idx == self.active_hand and self.play_state in {STATE_PLAYER_ACTION, STATE_DEAL}:
                glow = pygame.Rect(hx - 18, hy - 18, 240, CARD_H + 74)
                pygame.draw.rect(self.screen, (255, 220, 90, 90), glow, border_radius=16)
                pygame.draw.rect(self.screen, YELLOW, glow, width=3, border_radius=16)
            self.draw_hand(self.hands[idx], hx, hy)
            total_text = self.stat_font.render(self.total_text(self.hands[idx], f"Hand {idx + 1}"), True, WHITE)
            self.screen.blit(total_text, (hx, hy + CARD_H + 8))
            bet = self.bets.get(idx, 1.0)
            total_wagered += bet
            bet_text = self.small_font.render(f"Bet: {bet:.1f}", True, WHITE)
            self.screen.blit(bet_text, (hx, hy + CARD_H + 34))

        if self.highlight_action:
            action_color = ACTION_COLORS.get(self.highlight_action, WHITE)
            action_surf = self.action_font.render(f"ACTION: {self.highlight_action.upper()}", True, action_color)
            self.screen.blit(action_surf, (WIDTH // 2 - action_surf.get_width() // 2, 390))

        panel = pygame.Rect(22, HEIGHT - 168, 430, 138)
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=10)
        pygame.draw.rect(self.screen, GRAY, panel, width=1, border_radius=10)
        controls = [
            "Space → play/pause",
            "Right/Left → next/previous step",
            "+ / - → speed up / slow down",
            "[ / ] → previous/next episode",
        ]
        for i, text in enumerate(controls):
            self.screen.blit(self.small_font.render(text, True, WHITE), (38, HEIGHT - 155 + i * 30))

        stats_x = WIDTH - 310
        self.screen.blit(self.stat_font.render(f"Bet: {self.bets.get(self.active_hand, 1.0):.1f}", True, WHITE), (stats_x, HEIGHT - 160))
        self.screen.blit(self.stat_font.render(f"Total wagered: {total_wagered:.1f}", True, WHITE), (stats_x, HEIGHT - 130))
        self.screen.blit(self.stat_font.render(f"Net result: {self.total_reward:+.1f}", True, WHITE), (stats_x, HEIGHT - 100))
        self.screen.blit(
            self.small_font.render(f"State: {self.play_state} | Speed {self.speed:.1f}x | Episode {self.episode_idx + 1}/{len(self.episodes)}", True, WHITE),
            (stats_x - 100, HEIGHT - 68),
        )

        trace_y = 346
        for msg in self.dealer_trace[-3:]:
            self.screen.blit(self.small_font.render(msg, True, WHITE), (86, trace_y))
            trace_y += 22

        if self.outcomes:
            y = 112
            for out in self.outcomes:
                self.screen.blit(
                    self.small_font.render(
                        f"Hand {out['hand_index'] + 1}: {out['outcome']} ({out['reward']:+.1f})",
                        True,
                        WHITE,
                    ),
                    (WIDTH - 410, y),
                )
                y += 24

        if self.animation:
            now = pygame.time.get_ticks()
            elapsed = max(0, now - self.animation["start_time"])
            t = min(1.0, elapsed / self.animation["duration"])
            sx, sy = self.animation["start"]
            ex, ey = self.animation["end"]
            cx = int(sx + (ex - sx) * t)
            cy = int(sy + (ey - sy) * t)
            self.screen.blit(self.card_surface(self.animation["card"]), (cx, cy))

        if self.play_state == STATE_RESULT and self.result_overlay:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 110))
            self.screen.blit(overlay, (0, 0))
            color = GOLD if self.total_reward >= 0 else RED
            txt = self.action_font.render(self.result_overlay, True, color)
            self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - txt.get_height() // 2))

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT:
                        self.step_forward()
                    elif event.key == pygame.K_LEFT:
                        self.step_back()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.speed = min(4.0, self.speed * 2)
                    elif event.key == pygame.K_MINUS:
                        self.speed = max(0.25, self.speed / 2)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.episode_idx = (self.episode_idx + 1) % len(self.episodes)
                        self.reset_episode()
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.episode_idx = (self.episode_idx - 1) % len(self.episodes)
                        self.reset_episode()

            self.update()
            self.draw()
            self.clock.tick(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    replay_path = Path(args.replay)
    episodes = load_replay(replay_path)
    if not episodes:
        print("No episodes in replay file.")
        sys.exit(1)
    ReplayUI(episodes, speed=args.speed).run()


if __name__ == "__main__":
    main()
