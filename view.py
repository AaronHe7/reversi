import pygame, time
from pygame import gfxdraw
from reversi import Reversi

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
class View:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.game = Reversi()
        self.is_clicked = False
        self.end = False
        self.last_move = []
        self.in_menu = True
        self.player = -1
        self.computer = 1

    def display(self):
        pygame.draw.rect(self.screen, WHITE, (0, 0, self.width, self.height))
        self.display_board()
        self.display_stats()
        self.detect_win()
    
    def display_menu(self):
        pygame.draw.rect(self.screen, WHITE, (0, 0, self.width, self.height))
        large_font = pygame.font.SysFont("Arial", 40)
        title = large_font.render("Reversi", True, BLACK)
        self.screen.blit(title, ((self.width - title.get_width())/2, self.height * 0.1))
        font = pygame.font.SysFont("Arial", 30)
        color_text = large_font.render("Choose a color:", True, BLACK)
        self.screen.blit(color_text, ((self.width - color_text.get_width())/2, self.height * 0.3))
        c1 = [int(self.width * 0.4), int(self.height * 0.5)]
        c2 = [int(self.width * 0.6), int(self.height * 0.5)]
        r = 40
        pygame.gfxdraw.filled_circle(self.screen, c1[0], c1[1], r, BLACK)
        pygame.gfxdraw.aacircle(self.screen, c1[0], c1[1], r, BLACK)
        pygame.gfxdraw.aacircle(self.screen, c2[0], c2[1], r, BLACK)

    def display_board(self):
        game = self.game
        self.gridh = self.gridw = gridh = gridw = 500
        self.tile_size = tile_size = gridh/8
        self.gridx = x = (self.width - gridw)/2
        self.gridy = y = (self.height - gridh) * 0.8
        pygame.draw.rect(self.screen, GREEN, (x, y, gridw, gridh))
        valid_moves = self.game.get_moves()
        for i in range(9): # horizontal lines
            pygame.draw.line(self.screen, BLACK, (x, y + i * tile_size), (x + gridw, y + i * tile_size))
        for i in range(9): # horizontal lines
            pygame.draw.line(self.screen, BLACK, (x + i * tile_size, y), (x + i * tile_size, y + gridh))
        for i in range(8):
            for j in range(8):
                center = (int(tile_size/2 + x + tile_size * j), int(tile_size/2 + y + tile_size * i))
                r = int(tile_size/2 * 0.9)
                if [i, j] in valid_moves and (self.game.turn == self.player):
                    color = (255, 255, 255, 64) if game.turn == 1 else (0, 0, 0, 64)
                elif game.grid[i][j] == 1:
                    color = WHITE
                elif game.grid[i][j] == -1:
                    color = BLACK 
                else:
                    continue
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], r, color)
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], r, color)
                if [i, j] == self.last_move:
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(r/10), RED)
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(r/10), RED)
    
    def display_stats(self):
        r = 20
        font = pygame.font.SysFont("Arial", 30)
        textw = font.render(str(self.game.count[1]), True, BLACK)
        pygame.gfxdraw.aacircle(self.screen, int(self.width/5), int(self.gridy/2), r, BLACK)
        self.screen.blit(textw, (self.width/5 + 1.5 * r, self.gridy/2 - 15))
        textb = font.render(str(self.game.count[-1]), True, BLACK)
        pygame.gfxdraw.filled_circle(self.screen, int(2 * self.width/5), int(self.gridy/2), r, BLACK)
        pygame.gfxdraw.aacircle(self.screen, int(2 * self.width/5), int(self.gridy/2), r, BLACK)
        self.screen.blit(textb, (int(2/5 * self.width + 1.5 * r), int(self.gridy/2 - 15)))
    
    def declare_win(self, message):
        font = pygame.font.SysFont("Arial", 30)
        text = font.render(message, True, BLACK) 
        self.screen.blit(text, (int(0.55 * self.width), int(self.gridy/2 - 15)))

    def dist(self, coord1, coord2):
        return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2

    def register_click(self, x, y):
        if not self.in_menu:
            # if click is inside board
            if x >= self.gridx and y >= self.gridy and x <= self.gridx + self.gridw and y <= self.gridy + self.gridh:
                r = int((y - self.gridy)/self.tile_size)
                c = int((x - self.gridx)/self.tile_size)
                self.player_move(r, c)
        else:
            c1 = [int(self.width * 0.4), int(self.height * 0.5)]
            c2 = [int(self.width * 0.6), int(self.height * 0.5)]
            r = 40
            if self.dist([x, y], c1) < r * r:
                self.in_menu = False
                self.player = -1
                self.computer = 1
            if self.dist([x, y], c2) < r * r:
                self.in_menu = False
                self.player = 1
                self.computer = -1


    def player_move(self, r, c):
        if [r, c] in self.game.get_moves():
            self.game.make_move(r, c)
            self.last_move = [r, c]
    
    def computer_move(self, trials = 1000):
        self.last_move = self.game.computer_move("mcts", trials)
    
    def detect_win(self):
        if self.game.win(self.player):
            self.declare_win("Player wins!")
        elif self.game.win(self.computer):
            self.declare_win("Computer wins!")
        elif self.game.tie():
            self.declare_win("Tie!")

    def main(self):
        pygame.init()
        pygame.display.set_caption("Reversi")
        screen = self.screen = pygame.display.set_mode((self.width, self.height))
        screen.fill(WHITE)
        running = True
        game = self.game
        while running:
            if not self.in_menu and self.game.turn != self.player and not self.game.game_over():
                self.computer_move()
                self.display()
                pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.is_clicked:
                    self.is_clicked = True
                    self.register_click(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.is_clicked:
                    self.is_clicked = False
            self.display()
            if self.in_menu:
                self.display_menu()
            else:
                self.display()
            pygame.display.update()
