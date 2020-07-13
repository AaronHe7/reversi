import math, copy

class Node:
    def __init__(self, game, parent = None):
        self.game = game
        self.children = []
        self.parent = parent
        self.wins = 0
        self.trials = 0

    def value(self):
        return self.wins/self.trials + math.sqrt(2 * math.log(self.parent.trials/self.trials))

    def select(self):
        if self.game.game_over() or len(self.children) < len(self.game.get_moves()):
            return self
        best_val = -1
        best_child = 0
        for child in self.children:
            if child.value() > best_val:
                best_val = child.value()
                best_child = child
        return best_child.select()
    
    def expand(self):
        i = len(self.children)
        board = copy.deepcopy(self.game)
        board.make_move(board.get_moves()[i][0], board.get_moves()[i][1])
        self.children.append(Node(board, self))
        return self.children[i]
    
    def simulate(self):
        if self.game.count['b'] > self.game.count['w']:
            return 'b'
        elif self.game.count['w'] > self.game.count['b']:
            return 'w'
        else:
            return 't'
    
    def backpropagate(self, winner):
        self.trials += 1
        if self.parent:
            if winner == self.parent.game.turn:
                self.wins += 1
            elif winner == 't':
                self.wins += 0.5
            self.parent.backpropagate(winner)