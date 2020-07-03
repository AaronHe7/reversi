class Reversi: 
    def __init__(self):
        self.grid = [['' for i in range(8)] for i in range(8)]
        self.grid[3][3] = self.grid[4][4] = 'w'
        self.grid[3][4] = self.grid[4][3] = 'b'