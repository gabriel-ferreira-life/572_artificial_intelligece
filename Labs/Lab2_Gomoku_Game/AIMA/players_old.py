import copy
import itertools
import random
from collections import namedtuple
import numpy as np
import time


# Players for Games
move_times = []
move_scores = []
def alpha_beta_player(depth, evaluation_func):
    def alpha_beta_player_sub(game, state):
        start = time.time()
        action = alpha_beta_cutoff_search(state, game, depth, None, evaluation_func)
        move_times.append(time.time() - start)
        return action
    return alpha_beta_player_sub

# decision time
def get_average_decision_time():
    global move_times
    avg_move_time = np.mean(move_times) if move_times else float('inf')
    return round(avg_move_time, 2)

# number of moves
def reset_move_metrics():
    global move_times
    move_times = []
    
def get_player_move_times():
    return len(move_times.copy())+1


# move scores
def reset_score_metrics():
    global move_scores
    move_scores = []
    
def get_average_score():
    global move_scores
    return np.mean(move_scores) if move_scores else float('inf')

def get_move_scores():
    return move_scores.copy()


# human player
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


# from functools import lru_cache
# @lru_cache(maxsize=None) 
def alpha_beta_cutoff_search(state, game, d=None, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    
    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state, game)
        v = -np.inf
        sorted_actions = sorted(game.actions(state), key=lambda a: len(game.result(state, a).board) - len(state.board), reverse=True)
        for a in sorted_actions:
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state, game)
        v = np.inf
        sorted_actions = sorted(game.actions(state), key=lambda a: len(game.result(state, a).board) - len(state.board), reverse=True)
        for a in sorted_actions:
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
    move_scores.append(best_action) # store the scores
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
def count_patterns(state, player, pattern_length):
    count = 0
    board_size = len(state.board)
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]  # Four directions

    for x in range(board_size):
        for y in range(board_size):
            if state.board.get((x, y)) == player:
                pattern_found = True
                for dx, dy in directions:
                    for step in range(1, pattern_length):
                        nx, ny = x + dx * step, y + dy * step
                        if not (0 <= nx < board_size and 0 <= ny < board_size) or state.board.get((nx, ny)) != player:
                            pattern_found = False
                            break
                    if not pattern_found:
                        break  # Early stopping if pattern not found

                if pattern_found:
                    count += 1

    return count


def evaluate_game_state_improved(state, game):
    player = state.to_move
    opponent = 'O' if player == 'X' else 'X'
    score = 0

    # Basic score based on stone difference
    score += sum(1 for pos, occupant in state.board.items() if occupant == player) - sum(1 for pos, occupant in state.board.items() if occupant == opponent)

    # Pattern evaluation (combined loop)
    score += (count_patterns(state, player, 3) - count_patterns(state, opponent, 3)) * 3
    score += (count_patterns(state, player, 4) - count_patterns(state, opponent, 4)) * 4
    score += (count_patterns(state, player, 5) - count_patterns(state, opponent, 5)) * 100

    # Mobility
    player_mobility = len(game.actions(state))
    score += player_mobility

    # Strategic evaluations (simplified example)
    center_positions = [(len(state.board) // 2, len(state.board) // 2),
                      (len(state.board) // 2 - 1, len(state.board) // 2),
                      (len(state.board) // 2, len(state.board) // 2 - 1),
                      (len(state.board) // 2 + 1, len(state.board) // 2),
                      (len(state.board) // 2, len(state.board) // 2 + 1)]
    for pos in center_positions:
        if state.board.get(pos) == player:
            score += 20  # Bonus for controlling center positions

    # ... (consider more sophisticated position value calculation)

    return score



def get_score(state, game):
    total_score = 0
    already_existing_head_tails = set()
    score = {1: 10, 2: 100, 3: 1000, 4: 10000, 5: float('inf')}

    directions = [(1, 0), (0, 1)]
    current_player = state.to_move 
    for move, player in state.board.items():
        if player == current_player:
            for vector in directions:
                head = (move[0] + vector[0], move[1] + vector[1])
                tail = (move[0] - vector[0], move[1] - vector[1])
                cur_len = 1
                head_block = tail_block = False

                 # Expand head (early stopping)
                while state.board.get(head) == current_player and state.board.get(head[0] + vector[0], head[1] + vector[1]) != 'W':
                    head = (head[0] + vector[0], head[1] + vector[1])
                    cur_len += 1

                # Check head blocking
                if state.board.get(head) != 'W' and state.board.get(head) is not None:
                    head_block = True

                # Reset head to one step back
                head = (head[0] - vector[0], head[1] - vector[1])

                while state.board.get(tail) == current_player and state.board.get(tail[0] - vector[0], tail[1] - vector[1]) != 'W':
                    tail = (tail[0] - vector[0], tail[1] - vector[1])
                    cur_len += 1
                    
                # Check tail blocking
                if state.board.get(tail) != 'W' and state.board.get(tail) is not None:
                    tail_block = True

                # Reset tail to one step back
                tail = (tail[0] + vector[0], tail[1] + vector[1])

                headTail = (head, tail)
                if headTail not in already_existing_head_tails:
                    already_existing_head_tails.add(headTail)
                    if cur_len >= 5:
                        return score[5]
                    if (head_block and not tail_block) or (not head_block and tail_block):
                        if cur_len == 4:  # Forced win
                            total_score += 10000
                        else:
                            total_score += score[cur_len]
                    if not (head_block or tail_block):
                        if cur_len == 4: total_score += 1000
                        total_score += 2 * score[cur_len]

    return total_score
# ______________________________________________________________________________


