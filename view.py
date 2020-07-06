import pygame, time
from pygame import gfxdraw
from reversi import Reversi

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
class View:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.game = Reversi()
        self.is_clicked = False
        self.end = False

    def display(self):
        pygame.draw.rect(self.screen, WHITE, (0, 0, self.width, self.height))
        self.display_board()
        self.display_stats()

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
                    color = (255, 255, 255, 64) if game.turn == 'w' else (0, 0, 0, 64)
                elif game.grid[i][j] == 'w':
                    color = WHITE
                elif game.grid[i][j] == 'b':
                    color = BLACK 
                else:
                    continue
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], r, color)
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], r, color)
    
    def display_stats(self):
        r = 20
        font = pygame.font.SysFont("Arial", 30)
        textw = font.render(str(self.game.count['w']), False, BLACK)
        pygame.gfxdraw.aacircle(self.screen, int(self.width/5), int(self.gridy/2), r, BLACK)
        self.screen.blit(textw, (self.width/5 + 1.5 * r, self.gridy/2 - 15))
        textb = font.render(str(self.game.count['b']), False, BLACK)
        pygame.gfxdraw.filled_circle(self.screen, int(2 * self.width/5), int(self.gridy/2), r, BLACK)
        pygame.gfxdraw.aacircle(self.screen, int(2 * self.width/5), int(self.gridy/2), r, BLACK)
        self.screen.blit(textb, (int(2/5 * self.width + 1.5 * r), int(self.gridy/2 - 15)))

    def register_click(self, x, y):
        # if click is inside board
        if x >= self.gridx and y >= self.gridy and x <= self.gridx + self.gridw and y <= self.gridy + self.gridh:
            r = int((y - self.gridy)/self.tile_size)
            c = int((x - self.gridx)/self.tile_size)
            self.player_move(r, c)

    def player_move(self, r, c):
        if [r, c] in self.game.get_moves():
            self.game.make_move(r, c)
            if self.game.win('b'):
                self.game.end = True
                print("Black wins!")
            elif self.game.win('w'):
                print("White wins!")
            elif self.game.tie():
                print("Tie!")
    
    def computer_move(self):
        self.game.computer_move()
        if self.game.win('b'):
            self.game.end = True
            print("Black wins!")
        elif self.game.win('w'):
            self.game.end = True
            print("White wins!")
        elif self.game.tie():
            self.game.end = True
            print("Tie!")

    def main(self):
        pygame.init()
        pygame.display.set_caption("Reversi")
        screen = self.screen = pygame.display.set_mode((self.width, self.height))
        screen.fill(WHITE)
        running = True
        game = self.game
        self.player = 'b'
        self.display()
        pygame.display.update()
        while running:
            if self.game.turn != self.player and not self.game.end:
                self.computer_move()
                self.display()
                pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.is_clicked and (self.game.turn == self.player):
                    self.is_clicked = True
                    self.register_click(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.is_clicked:
                    self.is_clicked = False
            self.display()
            pygame.display.update()
