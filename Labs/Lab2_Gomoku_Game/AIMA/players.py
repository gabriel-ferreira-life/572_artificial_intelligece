import copy
import itertools
import random
from collections import namedtuple
import numpy as np



def alpha_beta_cutoff_search(state, game, d=None, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    
    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state, game)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state, game)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
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

# ______________________________________________________________________________
# # Heuristic Simple
def count_open_lines(game, state, player):
    return 0

def count_threat_spaces(game, state, player):
    return 0

def evaluate_game_state_simple(state, game):
    player = state.to_move
    opponent = 'W' if player == 'B' else 'B'
    score = 0
    
    # Count the number of player's stones vs opponent's stones
    player_stones = sum(1 for pos, occupant in state.board.items() if occupant == player)
    opponent_stones = sum(1 for pos, occupant in state.board.items() if occupant == opponent)
    score = player_stones - opponent_stones

    return score



# # Heuristic Improved
def count_patterns(state, player, pattern_type):
    count = 0
    board_size = len(state.board)
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]  # Horizontal, Vertical, Diagonal Right, Diagonal Left
    
    for x in range(board_size):
        for y in range(board_size):
            # Adjusted to use tuple keys for dictionary access
            if state.board.get((x, y)) == player:
                for dx, dy in directions:
                    pattern_length = 1
                    for step in range(1, 5):  # Look ahead up to 4 additional steps
                        nx, ny = x + dx * step, y + dy * step
                        if state.board.get((nx, ny)) == player:
                            pattern_length += 1
                        else:
                            break
                    
                    # Possible pattern
                    if pattern_type == 'open_three' and pattern_length == 3:
                        count += 1
                    elif pattern_type == 'closed_four' and pattern_length == 4:
                        count += 1
                    elif pattern_type == 'five' and pattern_length == 5:
                        count += 1
    return count

def position_value(pos):
    return 1


def evaluate_game_state_improved(state, game):
    player = state.to_move
    opponent = 'O' if player == 'X' else 'X'
    score = 0

    # Count stones for a basic score
    player_stones = sum(1 for pos, occupant in state.board.items() if occupant == player)
    opponent_stones = sum(1 for pos, occupant in state.board.items() if occupant == opponent)
    score += player_stones - opponent_stones

    # Incorporate pattern evaluation in the score
    score += (count_patterns(state, player, 'open_three') - count_patterns(state, opponent, 'open_three')) * 3
    score += (count_patterns(state, player, 'closed_four') - count_patterns(state, opponent, 'closed_four')) * 4
    score += (count_patterns(state, player, 'five') - count_patterns(state, opponent, 'five')) * 100  # Winning condition

    # Mobility calculation
    player_mobility = len(game.actions(state))

    score += player_mobility

    # Strategic evaluations
    if count_patterns(state, opponent, 'closed_four') > 0:
        score -= 50
    if count_patterns(state, player, 'closed_four') > 0 or count_patterns(state, player, 'open_three') > 1:
        score += 50

    # Position value calculation
    for pos, occupant in state.board.items():
        if occupant == player:
            score += position_value(pos)
        elif occupant == opponent:
            score -= position_value(pos)

    return score


# ______________________________________________________________________________
# Players for Games
def alpha_beta_player(depth, evaluation_func):
    def alpha_beta_player_sub(game, state):
        return alpha_beta_cutoff_search(state, game, depth, None, evaluation_func)
    return alpha_beta_player_sub


def human_player(game, state):
    print("Current board state:")
    game.display(state)
    print("List of legal moves at this state: {}".format(game.actions(state)))
    print("")
    move = None
    while move not in game.actions(state):
        try:
            move_input = input("Enter your move as 'row,column' (e.g. 1,1): ")
            move = tuple(int(x) for x in move_input.split(','))
        except ValueError:
            print("Invalid input. Please enter row and column as numbers separated by a comma.")
        if move not in game.actions(state):
            print("Invalid move. Please try again.")
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

