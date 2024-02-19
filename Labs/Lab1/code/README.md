# Lab1: The 8-puzzle

## Problem Description
In this programming assignment, you will write a code to solve 8-puzzle problems. The objective
of the puzzle is to rearrange a given initial configuration (starting state) of 8 numbers on a 3 x 3
board into a final configuration (goal state) with a minimum number of actions.

## Please note
All logic and instructions provided below assume that Part2.zip and Part3.zip have been extracted and are available in the directory structure as follows:

Lab1/
├── code/
│   └── (all logic files and notebooks)
├── Part2/
│   └── (all files/states for part 2 of the lab)
└── Part3/
    ├── L8/
    ├── L15/
    └── L24/

## Usage

### Components:

- output/: A directory that stores the outputs of part 3 of the lab.

- search_package.py: Contains the foundational classes and functions for all algorithms used in this lab. Portions of this code were sourced from the AIMA book repository on GitHub (https://github.com/aimacode) and modified to meet the specific requirements of this lab.

- part_1.py:  Implements the solution for part 1 of the lab. Utilizes classes and functions from search_package.py. To execute, open a console and run `%run part_1.py`. Follow the prompts to input the file path and desired algorithm. The printed output will display upon completion of the search. Applies to individual files.

- part_2.py: Implements the solution for part 2 of the lab, following a similar structure to `part_1.py` but applies to entire folder. Execute with `%run part_2.py` in the console, and follow the prompts.

- part_3.py: Addresses part 3 of the lab, automating the solution over multiple puzzles and levels. Running `%run part_3.py` in the console will:
1. Find 60 8-puzzles. 20 from each of 8, 15, and 24 levels, where the level indicates the optimal path length of the state from the goal.
2. Solve each puzzle and calculate the average run time and average nodes generated for all five algorithms across each level.
3. Output the results as tables in both .csv and .xlsx formats within the output folder.

- part_1_and_2.ipynb: A Jupyter notebook that mirrors the functionality of `part_1.py` and `part_2.py` but requires manual execution of all cells to generate output.

- part_3.ipynb: A Jupyter notebook that mirrors the functionality of `part_3.py` but requires manual execution of all cells to generate output.


## Acknowledgments
- [GitHub Pages] https://github.com/aimacode 
