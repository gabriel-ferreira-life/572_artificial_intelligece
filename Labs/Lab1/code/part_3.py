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
    tot_nodes_generated = len(explored) + len(frontier)

    # edn timing
    end_time = time.perf_counter()

    ## total time taken
    total_seconds = end_time - start_time



    return tot_nodes_generated, total_seconds, path_lenght, seq_actions
    
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

# Config
algorithms = ["BFGS", "IDS", "h1", "h2", "h3"]
index = [8, 15, 24]
final_df = pd.DataFrame()

# run code
for algo in algorithms:
    print("Solving for algo: ", algo)
    problem_levels = ["../Part3/L8/*.txt", "../Part3/L15/*.txt", "../Part3/L24/*.txt"]
    tot_nodes_generated_algo_avg = []
    time_taken_algo_avg = []

    for level in problem_levels:
        tot_nodes_generated_lvl_avg = []
        time_taken_lvl_avg = []

        for file in glob.glob(level):
            with open(file, 'r') as files:
                puzzle_raw = files.read().split()
                puzzle_int = tuple(int(x if x != '_' else '0') for x in puzzle_raw)


                tot_nodes_generated, time_taken, path_lenght, seq_actions = puzzle_8_solver(file, algo)
                tot_nodes_generated_lvl_avg.append(tot_nodes_generated)
                time_taken_lvl_avg.append(time_taken)


        tot_nodes_generated_algo_avg.append(np.mean(tot_nodes_generated_lvl_avg))
        time_taken_algo_avg.append(np.mean(time_taken_lvl_avg))

    
    data = {"Avg run time": time_taken_algo_avg, "Avg #nodes Explr": tot_nodes_generated_algo_avg}
    df_algo = pd.DataFrame(data, index=index)
    df_algo.columns = pd.MultiIndex.from_product([[algo], df_algo.columns])
    
    final_df = pd.concat([final_df, df_algo], axis=1)

final_df = final_df.reset_index().rename(columns={'index': "", "":"Depth"})
print(final_df)

final_df.to_excel("./output/performance_table.xlsx")
final_df.to_csv("./output/performance_table.csv", index=False)