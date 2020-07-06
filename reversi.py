import random

class Reversi: 
    def __init__(self):
        self.grid = [['' for i in range(8)] for i in range(8)]
        self.grid[3][3] = self.grid[4][4] = 'w'
        self.grid[3][4] = self.grid[4][3] = 'b'
        self.turn = 'b'
        self.end = False
        self.count = {'b': 2, 'w': 2}
        self.recompute_moves()

    def get_moves(self, player = None):
        if (player == None):
            player = self.turn
        return self.moves[player]
        
    def recompute_moves(self):
        self.moves = {'w': [], 'b': []}
        for player in "wb":
            for i in range(8):
                for j in range(8):
                    if self.valid_move(i, j, player):
                        self.moves[player].append([i, j])

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
        self.count[player] += 1
        swapped = []
        for d in range(8):
            nr = r + dy[d]
            nc = c + dx[d]
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8 or grid[nr][nc] == '' or grid[nr][nc] == player:
                continue
            valid_dir = False
            todo = []
            while nr >= 0 and nr < 8 and nc >= 0 and nc < 8 and grid[nr][nc] != '':
                if grid[nr][nc] == player:
                    valid_dir = True
                    break
                todo.append([nr, nc])
                nr += dy[d]
                nc += dx[d]
            if valid_dir:
                other_player = 'b' if player == 'w' else 'w'
                self.count[player] += len(todo)
                self.count[other_player] -= len(todo)
                for move in todo:
                    assert(grid[move[0]][move[1]] != player)
                    grid[move[0]][move[1]] = player
                    swapped.append(move)
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.recompute_moves()
        if not len(self.get_moves()):
            self.turn = 'b' if self.turn == 'w' else 'w'
            self.recompute_moves()
        return swapped
    
    def random_move(self):
        moves = self.get_moves()
        if len(moves) == 0:
            return
        move = moves[random.randint(0, len(moves) - 1)]
        self.make_move(move[0], move[1])
    

    def computer_move(self, depth = 3):
        player = self.turn
        moves = self.get_moves()
        if len(moves) == 0:
            return
        #move = moves[random.randint(0, len(moves) - 1)]
        move = self.minimax(True, None, depth)
        self.make_move(move["move"][0], move["move"][1])

    def minimax(self, maximizer = True, player = None, depth = 3):
        if player == None:
            player = self.turn
        other_player = 'b' if player == 'w' else 'w'
        maximizing_player = player if maximizer else other_player
        minimizing_player = 'w' if maximizing_player == 'b' else 'b' 
        best_move = {"score": (-65 if maximizer else 65), "move": []}
        # check for terminal state
        if self.win(maximizing_player):
            best_move["score"] = 64
            return best_move
        elif self.win(minimizing_player):
            best_move["score"] = -64
            return best_move
        elif self.tie():
            best_move["score"] = 0
            return best_move
        elif depth == 0:
            best_move["score"] = self.count[maximizing_player] - self.count[minimizing_player]
            return best_move

        moves = self.get_moves(player)
        gridbefore = self.grid.copy()
        for move in moves:
            cntb = self.count['b']
            cntw = self.count['w']
            replace = self.make_move(move[0], move[1])
            evaluation = self.minimax(maximizing_player == self.turn, self.turn, depth - 1)
            if maximizer:
                if evaluation["score"] > best_move["score"]:
                    best_move["move"] = move
                    best_move["score"] = evaluation["score"]
            else:
                if evaluation["score"] < best_move["score"]:
                    best_move["move"] = move
                    best_move["score"] = evaluation["score"]
            self.count['b'] = cntb
            self.count['w'] = cntw
            self.grid[move[0]][move[1]] = ''
            for cell in replace:
                self.grid[cell[0]][cell[1]] = other_player
            self.turn = player
        return best_move
    
    def game_over(self):
        return not len(self.get_moves('b')) and not len(self.get_moves('w'))

    def tie(self): 
        return self.count['b'] == self.count['w'] and self.game_over()
    
    def win(self, player):
        return self.game_over() and self.count[player] > self.count['w' if player == 'b' else 'b']