  # Libraries
import sys
from collections import deque
import heapq
import math

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
from contextlib import contextmanager
import signal
import time

# Needed to hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")

# Search Package
from search_package import *

# Part 1
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def time_config(total_seconds):
    # Calculate minutes, seconds, and microseconds
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    microseconds = int((total_seconds - int(total_seconds)) * 1_000_000)
    
    if (minutes == 0) and (seconds == 0):
        time_taken = f"{microseconds} microSec."
    
    elif minutes == 0:
        time_taken = f"{seconds} sec {microseconds} microSec."
    
    else:
        time_taken = f"{minutes} min {seconds} sec {microseconds} microSec."
        
    return time_taken

# define function to get the required output
def func_output(algo, algorithm_name, problem, heuristic=None, display=True):

    # start timing
    start_time = time.perf_counter()
    

    if algorithm_name == "BFGS" or algorithm_name == "BFTS" or algorithm_name == "IDS":
#         print(algorithm_name)
        solution, explored = algo(problem)
        frontier = []
    else:
        solution, explored, frontier = algo(problem)


    # required output
    seq_actions = solution.solution()
    path = solution.path()
    path_lenght = len(path)
    try:
        tot_nodes_generated = len(explored) + len(frontier)
    except:
        tot_nodes_generated = explored + len(frontier)

    # edn timing
    end_time = time.perf_counter()

    ## total time taken
    total_seconds = end_time - start_time

    # Calculate minutes, seconds, and microseconds
    time_taken = time_config(total_seconds)

    return print(f"Total nodes generated: {tot_nodes_generated}\n"
      f"Total Time Taken: {time_taken}\n"
      f"Path length: {path_lenght}\n"
      f"Path: {''.join(seq_actions)}")
    
def puzzle_8_solver(file_path, algorithm):
    try:
        with time_limit(900):  # 900 seconds = 15 minutes    
            
            # read files in
            with open(file_path, 'r') as file:
                puzzle_raw = file.read().split()
            puzzle_int = tuple(int(x if x != '_' else '0') for x in puzzle_raw)

            # fit puzzle in
            puzzle = EightPuzzle(puzzle_int)
#             puzzle = EightPuzzle((2, 4, 3, 1, 5, 6, 7, 8, 0))

            # check for solvability
            is_solvable = puzzle.check_solvability(puzzle_int)

            if is_solvable == False:
                print("Problem is not solvable.")
                return None 

            # dictionary to map algorithm names to their corresponding functions
            algo_dict = {
                'BFGS': breadth_first_graph_search,
                'BFTS': breadth_first_tree_search,
                'IDS': iterative_deepening_search,
                'h1': astar_search_1,
                'h2': astar_search_2,
                'h3': astar_search_3
            }


            if algorithm in algo_dict:
                return func_output(algo_dict[algorithm], algorithm, puzzle)
            else:
                print(f"Algorithm {algorithm} is not recognized. The available algorithms are: BFGS, BFTS, IDS, h1, h2, h3")

    except TimeoutException as e:
        print("Total nodes generated: Timed out")
        print("Total Time Taken: >15 min")
        print("Path length: Timed out")
        print("Path: Timed out")

# request user input
print("Insert file path and desired algorithm (BFGS, BFTS, IDS, h1, h2, h3).")
file_path = input("File path: ")
algorithm = input("Algorithm: ")
print("")

# calling Function
puzzle_8_solver(file_path, algorithm)