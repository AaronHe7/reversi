import pygame
from reversi import Reversi

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)

class View:
    def __init__(self, width, height):
        self.height = height
        self.width = width

    def __init__(self):
        self.height = self.width = 600

    def display(self, game):
        gridh = gridw = 500
        tile_size = gridh/8
        x = (self.width - gridw)/2
        y = (self.height - gridh)/2
        pygame.draw.rect(self.screen, GREEN, (x, y, gridw, gridh))
        for i in range(9): # horizontal lines
            pygame.draw.line(self.screen, BLACK, (x, y + i * tile_size), (x + gridw, y + i * tile_size))
        for i in range(9): # horizontal lines
            pygame.draw.line(self.screen, BLACK, (x + i * tile_size, y), (x + i * tile_size, y + gridh))
        for i in range(8):
            for j in range(8):
                if game.grid[i][j] == 'w':
                    color = WHITE
                elif game.grid[i][j] == 'b':
                    color = BLACK 
                if game.grid[i][j] != '':
                    pygame.draw.circle(self.screen, color, (int(x + i * tile_size + tile_size/2), int(y + j * tile_size + tile_size/2)), int(tile_size/2 * 0.9))

    def main(self):
        pygame.init()
        pygame.display.set_caption("Reversi")
        screen = self.screen = pygame.display.set_mode((self.width, self.height))
        screen.fill(WHITE)
        running = True
        game = Reversi()
        while running:
            self.display(game)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.update()

view = View()
view.main()