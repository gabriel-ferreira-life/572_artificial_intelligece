# Libraries
import numpy as np
import pandas as pd
import sys
from collections import deque
import heapq
import math
import glob

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
from contextlib import contextmanager
import signal
import time

import warnings
warnings.filterwarnings("ignore")

# Search Package
from search_package import *



# Part 3

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
    path_lenght = len(path) - 1
    
    try:
        tot_nodes_generated = len(explored) + len(frontier)
    except:
        tot_nodes_generated = explored + len(frontier)
    
    # edn timing
    end_time = time.perf_counter()

    ## total time taken
    total_seconds = end_time - start_time



    return tot_nodes_generated, total_seconds#, path_lenght, seq_actions
    
def puzzle_8_solver(file_path, algorithm):
            
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
        return None, None

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


# Config
algorithms = ["BFGS", "IDS", "h1", "h2", "h3"]
# algorithms = ["h2", "h3", "IDS"]
index = [8, 15, 24]
final_df = pd.DataFrame()

# run code
for algo in algorithms:
    print(" ")
    print("Solving for algo: ", algo)
    problem_levels = ["../Part3/L8/*.txt", "../Part3/L15/*.txt", "../Part3/L24/*.txt"]
    tot_nodes_generated_algo_avg = []
    time_taken_algo_avg = []

    for level in problem_levels:
        print(" ")
        print("At level: ", level)
        tot_nodes_generated_lvl_avg = []
        time_taken_lvl_avg = []

        for file in glob.glob(level):
            print("Processing file: ", file)
            with open(file, 'r') as files:
                puzzle_raw = files.read().split()
                puzzle_int = tuple(int(x if x != '_' else '0') for x in puzzle_raw)


                tot_nodes_generated, time_taken = puzzle_8_solver(file, algo)
                tot_nodes_generated_lvl_avg.append(tot_nodes_generated)
                time_taken_lvl_avg.append(time_taken)


        tot_nodes_generated_algo_avg.append(np.mean(tot_nodes_generated_lvl_avg))
        time_taken_algo_avg.append(np.mean(time_taken_lvl_avg))

    
    data = {"Avg run time": time_taken_algo_avg, "Avg #nodes Explr": tot_nodes_generated_algo_avg}
    df_algo = pd.DataFrame(data, index=index)
    df_algo.columns = pd.MultiIndex.from_product([[algo], df_algo.columns])
    
    final_df = pd.concat([final_df, df_algo], axis=1)
    final_df.to_csv("./output/performance_table.csv", index=False)
    

final_df = final_df.reset_index().rename(columns={'index': "", "":"Depth"})
print(final_df)

final_df.to_excel("./output/performance_table.xlsx")
final_df.to_csv("./output/performance_table.csv", index=False)