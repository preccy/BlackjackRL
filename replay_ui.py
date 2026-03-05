from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pygame

from replay_logger import load_replay

WIDTH, HEIGHT = 1200, 760
GREEN = (21, 102, 52)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GOLD = (214, 174, 66)
GRAY = (150, 150, 150)
RED = (210, 50, 50)
BLUE = (60, 120, 210)

BUTTONS = ["Hit", "Stand", "Double", "Split"]


class ReplayUI:
    def __init__(self, episodes: list[dict], speed: float = 1.0):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Blackjack Replay")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.small = pygame.font.SysFont("arial", 18)

        self.episodes = episodes
        self.episode_idx = 0
        self.events = []
        self.event_ptr = 0
        self.playing = True
        self.speed = speed
        self.last_tick = 0
        self.delay_ms = 250
        self.highlight_action = None

        self.reset_episode()

    def reset_episode(self):
        episode = self.episodes[self.episode_idx]
        self.events = episode.get("events", [])
        self.event_ptr = 0
        self.dealer = []
        self.hands: dict[int, list[dict]] = {0: []}
        self.bets: dict[int, float] = {0: 1.0}
        self.outcomes = []
        self.active_hand = 0
        self.highlight_action = None

    def card_surface(self, card: dict) -> pygame.Surface:
        surf = pygame.Surface((72, 104), pygame.SRCALPHA)
        pygame.draw.rect(surf, WHITE, (0, 0, 72, 104), border_radius=8)
        pygame.draw.rect(surf, BLACK, (0, 0, 72, 104), width=2, border_radius=8)
        txt = self.font.render(f"{card['rank']}{card['suit']}", True, RED if card["suit"] in ["♥", "♦"] else BLACK)
        surf.blit(txt, (8, 8))
        return surf

    def draw_button(self, text: str, x: int, y: int, w: int, h: int, active: bool = False):
        color = BLUE if active else GRAY
        pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, WHITE, (x, y, w, h), width=2, border_radius=10)
        label = self.small.render(text, True, WHITE)
        self.screen.blit(label, (x + w // 2 - label.get_width() // 2, y + h // 2 - label.get_height() // 2))

    def layout_hand_pos(self, area_y: int, hand_idx: int, total_hands: int):
        spacing = 240
        start_x = WIDTH // 2 - ((total_hands - 1) * spacing) // 2
        return start_x + hand_idx * spacing, area_y

    def step_event(self):
        if self.event_ptr >= len(self.events):
            return
        evt = self.events[self.event_ptr]
        self.event_ptr += 1
        t = evt.get("type")
        if t == "deal_card":
            if evt["to"] == "dealer":
                self.dealer.append(evt["card"])
            else:
                idx = evt["hand_index"]
                self.hands.setdefault(idx, [])
                self.hands[idx].append(evt["card"])
        elif t == "action":
            self.active_hand = evt.get("hand_index", 0)
            self.highlight_action = evt.get("action_name", "").capitalize()
            if self.highlight_action == "Stand":
                self.highlight_action = "Stand"
        elif t == "hand_transition":
            self.active_hand = evt.get("next_hand_index", 0)
        elif t == "final_result":
            self.outcomes = evt.get("outcomes", [])
            for out in self.outcomes:
                self.bets[out["hand_index"]] = out.get("bet", 1.0)

    def draw(self):
        self.screen.fill(GREEN)
        pygame.draw.ellipse(self.screen, (15, 85, 43), (80, 80, WIDTH - 160, HEIGHT - 160))

        dealer_lbl = self.font.render("Dealer", True, GOLD)
        self.screen.blit(dealer_lbl, (80, 70))
        for i, card in enumerate(self.dealer):
            surf = self.card_surface(card)
            self.screen.blit(surf, (260 + i * 82, 130))

        player_lbl = self.font.render("Player", True, GOLD)
        self.screen.blit(player_lbl, (80, 430))

        total_hands = max(1, len(self.hands))
        for idx in sorted(self.hands.keys()):
            hx, hy = self.layout_hand_pos(500, idx, total_hands)
            if idx == self.active_hand:
                pygame.draw.rect(self.screen, GOLD, (hx - 16, hy - 16, 200, 140), width=3, border_radius=10)
            for c_idx, card in enumerate(self.hands[idx]):
                self.screen.blit(self.card_surface(card), (hx + c_idx * 78, hy))
            bet_txt = self.small.render(f"Bet: {self.bets.get(idx,1.0):.1f}", True, WHITE)
            self.screen.blit(bet_txt, (hx, hy + 112))

        bx, by = WIDTH - 440, HEIGHT - 90
        for i, name in enumerate(BUTTONS):
            self.draw_button(name, bx + i * 105, by, 95, 48, active=self.highlight_action == name)

        ctrl = self.small.render(
            f"Space Play/Pause | Right Step | +/- Speed ({self.speed:.1f}x) | [ / ] Episode ({self.episode_idx+1}/{len(self.episodes)})",
            True,
            WHITE,
        )
        self.screen.blit(ctrl, (20, HEIGHT - 30))

        if self.outcomes:
            y = 360
            for out in self.outcomes:
                txt = self.small.render(
                    f"Hand {out['hand_index']}: {out['outcome']} reward {out['reward']:+.1f}", True, WHITE
                )
                self.screen.blit(txt, (80, y))
                y += 24

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
                        self.step_event()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.speed = min(4.0, self.speed * 2)
                    elif event.key == pygame.K_MINUS:
                        self.speed = max(0.5, self.speed / 2)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.episode_idx = (self.episode_idx + 1) % len(self.episodes)
                        self.reset_episode()
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.episode_idx = (self.episode_idx - 1) % len(self.episodes)
                        self.reset_episode()

            now = pygame.time.get_ticks()
            if self.playing and now - self.last_tick > self.delay_ms / self.speed:
                self.step_event()
                self.last_tick = now

            self.draw()
            self.clock.tick(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    episodes = load_replay(args.replay)
    if not episodes:
        print("No episodes in replay file.")
        sys.exit(1)
    ReplayUI(episodes, speed=args.speed).run()


if __name__ == "__main__":
    main()
