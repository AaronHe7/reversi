import random, copy, time, mcts

class Reversi: 
    def __init__(self):
        self.grid = [['' for i in range(8)] for i in range(8)]
        self.grid[3][3] = self.grid[4][4] = 'b'
        self.grid[3][4] = self.grid[4][3] = 'w'
        self.turn = 'b'
        self.count = {'b': 2, 'w': 2}
        self.recompute_moves()
        # t1 = time.clock()
        # sim = 300
        # for i in range(sim):
            # self.simulate()
        # t2 = time.clock()
        # print((t2 - t1)/sim)

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

    def make_move(self, r, c): # performs move, returns all swapped pieces
        grid = self.grid
        player = self.turn
        # assert(self.valid_move(r, c, player))
        dx = [-1, -1, -1, 1, 1, 1, 0, 0]
        dy = [1, 0, -1, 1, 0, -1, 1, -1]
        grid[r][c] = player
        self.count[player] += 1
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
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.recompute_moves()
        if not len(self.get_moves()):
            self.turn = 'b' if self.turn == 'w' else 'w'
            self.recompute_moves()
    
    def random_move(self):
        moves = self.get_moves()
        if len(moves) == 0:
            return
        move = moves[random.randint(0, len(moves) - 1)]
        self.make_move(move[0], move[1])
    
    def computer_move(self, algorithm, depth):
        player = self.turn
        moves = self.get_moves()
        if len(moves) == 0:
            return
        if algorithm == "minimax":
            move = self.minimax(True, depth)
            self.make_move(move["move"][0], move["move"][1])
            return move["move"]
        else:
            move = self.mcts(depth)
            self.make_move(move[0], move[1])
            return move

    def minimax(self, maximizing_player, depth):
        player = self.turn
        other_player = 'b' if player == 'w' else 'w'
        maximizer = player if maximizing_player else other_player
        minimizer = 'w' if maximizer == 'b' else 'b' 
        best_move = {"score": (-65 if maximizing_player else 65), "move": []}
        # check for terminal state
        if self.win(maximizer):
            best_move["score"] = 64
            return best_move
        elif self.win(minimizer):
            best_move["score"] = -64
            return best_move
        elif self.tie():
            best_move["score"] = 0
            return best_move
        elif depth == 0:
            best_move["score"] = self.count[maximizer] - self.count[minimizer]
            return best_move

        moves = self.get_moves(player)
        for move in moves:
            new_board = copy.deepcopy(self)
            new_board.make_move(move[0], move[1])
            evaluation = new_board.minimax(maximizer == new_board.turn, depth - 1)
            if maximizing_player:
                if evaluation["score"] > best_move["score"]:
                    best_move["move"] = move
                    best_move["score"] = evaluation["score"]
            else:
                if evaluation["score"] < best_move["score"]:
                    best_move["move"] = move
                    best_move["score"] = evaluation["score"]
        return best_move
    
    def mcts(self, simulations):
        evaluator = mcts.Mcts(self, True)
        return evaluator.best_move(simulations)

    def game_over(self):
        return not len(self.get_moves('b')) and not len(self.get_moves('w'))

    def tie(self): 
        return self.count['b'] == self.count['w'] and self.game_over()
    
    def win(self, player):
        return self.game_over() and self.count[player] > self.count['w' if player == 'b' else 'b']
