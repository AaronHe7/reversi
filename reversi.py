class Reversi: 
    def __init__(self):
        self.grid = [['' for i in range(8)] for i in range(8)]
        self.grid[3][3] = self.grid[4][4] = 'w'
        self.grid[3][4] = self.grid[4][3] = 'b'
        self.turn = 'b'

    def get_moves(self, player = None):
        if (player == None):
            player = self.turn
        moves = []
        for i in range(8):
            for j in range(8):
                if self.valid_move(i, j, player):
                    moves.append([i, j])
        return moves

    def valid_move(self, r, c, player = None):
        grid = self.grid
        if grid[r][c] != '':
            return False
        if (player == None):
            player = self.turn
        dx = [-1, -1, -1, 1, 1, 1, 0, 0]
        dy = [1, 0, -1, 1, 0, -1, 1, -1]
        for d in range(8):
            nr = r + dy[d]
            nc = c + dx[d]
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                continue
            if grid[nr][nc] == '' or grid[nr][nc] == player:
                continue
            while nr >= 0 and nr < 8 and nc >= 0 and nc < 8 and grid[nr][nc] != '':
                if grid[nr][nc] == player:
                    return True
                nr += dy[d]
                nc += dx[d]
        return False

    def make_move(self, r, c, player = None): # performs move, returns all swapped pieces
        grid = self.grid
        if (player == None):
            player = self.turn
        assert(self.valid_move(r, c, player))
        dx = [-1, -1, -1, 1, 1, 1, 0, 0]
        dy = [1, 0, -1, 1, 0, -1, 1, -1]
        grid[r][c] = player
        swapped = []
        for d in range(8):
            nr = r + dy[d]
            nc = c + dx[d]
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8 or grid[nr][nc] == '' or grid[nr][nc] == player:
                continue
            nr += dy[d]
            nc += dx[d]
            valid_dir = False
            while nr >= 0 and nr < 8 and nc >= 0 and nc < 8 and grid[nr][nc] != '':
                if grid[nr][nc] == player:
                    valid_dir = True
                    break
                nr += dy[d]
                nc += dx[d]
            if valid_dir:
                nr = r + dy[d]
                nc = c + dx[d]
                while nr >= 0 and nr < 8 and nc >= 0 and nc < 8 and grid[nr][nc] != '':
                    if grid[nr][nc] == player:
                        break
                    swapped.append([nr, nc])
                    grid[nr][nc] = player
                    nr += dy[d]
                    nc += dx[d]
        return swapped