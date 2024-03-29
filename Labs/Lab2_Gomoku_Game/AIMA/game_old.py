import copy
import itertools
import random
from collections import namedtuple
import numpy as np

GameState = namedtuple('GameState', 'to_move, utility, board, moves')
StochasticGameState = namedtuple('StochasticGameState', 'to_move, utility, board, moves, chance')



class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)
    


    def play_game(self, player1, player2):
        """Play an n-person, move-alternating game."""

        # Assign colors based on argument position
        self.players = { 'B': player1, 'W': player2 }

        state = self.initial
        while True:
            # Determine the current player ('B' or 'W')
            current_player = self.to_move(state)
            
            # Fetch the player function based on the current player's role
            player_func = self.players[current_player]
            
            # Call the player function to get the move
            move = player_func(self, state)
            
            # Apply the move to get the new state
            state = self.result(state, move)
            
            # Check if the game has reached a terminal state
            if self.terminal_test(state):
                self.display(state)

                # Determine and display the game outcome
                if self.utility(state, 'B') > 0:
                    print("Black wins!")
                    return 1
      
                elif self.utility(state, 'B') < 0:
                    print("White wins!")
                    return -1
  
                else:
                    print("It's a draw.")
                    return 0

                
                

class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'B'.
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'B' or 'W'."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='B', utility=0, board={}, moves=moves)

    def actions(self, state):
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('W' if state.to_move == 'B' else 'B'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'B' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'B' wins with this move, return 1; if 'W' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'B' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k

    
    
class Gomoku(TicTacToe):
    def __init__(self, h=15, v=15, k=5):
        super().__init__(h, v, k)
        center = ((h // 2) + 1, (v // 2) + 1)
        
        # Initialize the board with the first move at the center
        self.initial = GameState(to_move='W', utility=0, board={center: 'B'}, moves=self.calculate_initial_moves(center))

    def calculate_initial_moves(self, center):
        moves = [(x, y) for x in range(1, self.h + 1) for y in range(1, self.v + 1)]
        moves.remove(center)
        return moves

    def actions(self, state):
        if list(state.board.values()).count('B') == 1 and state.to_move == 'B':
            center = ((self.h // 2) + 1, (self.v // 2) + 1)
            return [(x, y) for x in range(1, self.h + 1) for y in range(1, self.v + 1)
                    if max(abs(x - center[0]), abs(y - center[1])) >= 3 and (x, y) not in state.board]
        else:
            # Standard action generation, excluding occupied positions
            return [(x, y) for x in range(1, self.h + 1) for y in range(1, self.v + 1) if (x, y) not in state.board]