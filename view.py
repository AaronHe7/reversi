import pygame
from reversi import Reversi

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
class View:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.game = Reversi()
        self.valid_moves = self.game.get_moves()
        self.is_clicked = False

    def display_board(self):
        game = self.game
        self.gridh = self.gridw = gridh = gridw = 500
        self.tile_size = tile_size = gridh/8
        self.gridx = x = (self.width - gridw)/2
        self.gridy = y = (self.height - gridh)/2
        pygame.draw.rect(self.screen, GREEN, (x, y, gridw, gridh))
        for i in range(9): # horizontal lines
            pygame.draw.line(self.screen, BLACK, (x, y + i * tile_size), (x + gridw, y + i * tile_size))
        for i in range(9): # horizontal lines
            pygame.draw.line(self.screen, BLACK, (x + i * tile_size, y), (x + i * tile_size, y + gridh))
        for i in range(8):
            for j in range(8):
                center = (int(tile_size/2), int(tile_size/2))
                r = int(tile_size/2 * 0.9)
                surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA).convert_alpha()
                if [i, j] in self.valid_moves:
                    color = (255, 255, 255, 64) if game.turn == 'w' else (0, 0, 0, 64)
                elif game.grid[i][j] == 'w':
                    color = WHITE
                elif game.grid[i][j] == 'b':
                    color = BLACK 
                else:
                    continue
                pygame.draw.circle(surface, color, center, r) 
                self.screen.blit(surface, (int(x + i * tile_size), int(y + j * tile_size)))
    
    def register_click(self, x, y):
        # if click is inside board
        if x >= self.gridx and y >= self.gridy and x <= self.gridx + self.gridw and y <= self.gridy + self.gridh:
            r = int((x - self.gridx)/self.tile_size)
            c = int((y - self.gridy)/self.tile_size)
            print(str(r) + ' ' + str(c))
            if [r, c] in self.valid_moves:
                self.game.make_move(r, c)
                self.game.turn = 'b' if self.game.turn == 'w' else 'w'
                self.valid_moves = self.game.get_moves()

    def main(self):
        pygame.init()
        pygame.display.set_caption("Reversi")
        screen = self.screen = pygame.display.set_mode((self.width, self.height))
        screen.fill(WHITE)
        running = True
        game = self.game
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.is_clicked:
                    self.is_clicked = True
                    self.register_click(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.is_clicked:
                    self.is_clicked = False
            self.display_board()
            pygame.display.update()