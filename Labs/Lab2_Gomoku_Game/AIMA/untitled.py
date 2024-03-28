import random
from collections import namedtuple
import numpy as np

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

def alpha_beta_cutoff_search(state, game, d=3, cutoff_test=None, eval_fn=None):

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action
# __________________________
# Players for Games
def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("Available moves: {}".format(game.actions(state)))
    print("")

    move = None
    while move is None:
        if not game.actions(state):
            print('No legal moves: passing turn to next player')
            return None

        move_string = input('Your move? ')

        # Example of handling input without eval(), assuming moves can be input as comma-separated values
        try:
            # Assuming moves are tuples like (row, col), input should be formatted as "row, col"
            move_parts = move_string.strip().split(',')
            if len(move_parts) == 2:  # Adjust based on expected format
                move = tuple(int(part.strip()) for part in move_parts)
                # Validate if move is in the list of available actions
                if move not in game.actions(state):
                    print("Invalid move. Please try again.")
                    move = None  # Reset move to None to continue the loop
            else:
                print("Invalid format. Please enter the move as 'row, col'.")
                move = None
        except ValueError:
            print("Invalid input. Please ensure you enter numeric values for row and column.")
            move = None
        except Exception as e:
            print(f"Unexpected error: {e}")
            move = None

    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

def alpha_beta_player(game, state):
    return alpha_beta_cutoff_search(state, game)

class Gomoku():
    """Also known as Five in a row."""

    def _init_(self, h=15, v=16, k=5):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""

    def actions(self, state):
            if state.move_count == 0:
                # Black's first move, must be at the center
                return [(self.h // 2, self.v // 2)]
            elif state.move_count == 1:
                # White's first move, can be anywhere
                return [(x, y) for x in range(self.h) for y in range(self.v) if state.board[x][y] == '.']
            elif state.move_count == 2:
                # Black's second move, must be at least three intersections away from the first stone
                center = self.h // 2
                return [(x, y) for x in range(self.h) for y in range(self.v)
                        if state.board[x][y] == '.' and (abs(x - center) >= 3 or abs(y - center) >= 3)]
            else:
                # Any other move
                return [(x, y) for x in range(self.h) for y in range(self.v) if state.board[x][y] == '.']

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    # def get_legal_moves(board, player, move_count):
    #     size = len(board)
    #     legal_moves = []
    #     if move_count == 0 and player == 'black':
    #         # First move of the game; black places in the center.
    #         return [(size // 2, size // 2)]
    #     elif move_count == 1 and player == 'white':
    #         # White's first move can be anywhere.
    #         legal_moves = [(r, c) for r in range(size) for c in range(size) if board[r][c] == '.']
    #     elif move_count == 2:
    #         # Black's second move must be at least three intersections away from the first stone.
    #         center = size // 2
    #         for r in range(size):
    #             for c in range(size):
    #                 if board[r][c] == '.' and (abs(r - center) >= 3 or abs(c - center) >= 3):
    #                     legal_moves.append((r, c))
    #     else:
    #         # All other moves can be anywhere on the board.
    #         legal_moves = [(r, c) for r in range(size) for c in range(size) if board[r][c] == '.']
    #     return legal_moves

    def _repr_(self):
        return '<{}>'.format(self._class.name_)

    def compute_utility(self, board, move, player):
            """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
            if (self.k_in_row(board, move, player, (0, 1)) or
                    self.k_in_row(board, move, player, (1, 0)) or
                    self.k_in_row(board, move, player, (1, -1)) or
                    self.k_in_row(board, move, player, (1, 1))):
                return +1 if player == 'X' else -1
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

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))
    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

def display_gomoku(current_state):
    pass


def play_detailed_gomoku():
    # Introduction
    print("Welcome to Gomoku!")
    print("You are Player 1 ('X'), and you are playing against an AI ('O').")
    print("To make a move, enter the row and column numbers separated by a comma (e.g., 3,5).")
    print("Let's start the game!")
    print("--------------------------------------------------")

    # Initialize the game
    game = Gomoku()

    # Define players
    player1 = query_player  # Human player
    player2 = alpha_beta_player  # AI player

    # Variable to store the game state after each move
    current_state = game.initial

    # Loop to play the game
    while not game.terminal_test(current_state):
        # Display the board
        display_gomoku(current_state)

        if current_state.to_move == 'X':
            print("Your turn, Player 1 ('X').")
            move = player1(game, current_state)
        else:
            print("AI is making its move...")
            move = player2(game, current_state)

        # Apply the move
        current_state = game.result(current_state, move)

    # Game over, display the final board
        display_gomoku(current_state)

    # Display and integrate results
    result = game.utility(current_state, game.to_move(game.initial))
    print("--------------------------------------------------")
    print("Game Over!")
    if result > 0:
        print("Congratulations, Player 1 ('X'), you've won the game!")
    elif result < 0:
        print("The AI ('O') has won the game. Better luck next time!")
    else:
        print("It's a draw. Well played both!")


if _name_ == "_main_":
    play_detailed_gomoku()