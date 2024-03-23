# Lab 2: Gomoku Game Implementation and Analysis

This repository contains a Python implementation of the Gomoku game (also known as Five in a Row), along with utilities for playing the game, implementing various player strategies, and analyzing gameplay. Below is an overview of each file and its role in the project.

## Overview

**game.py:** Defines the core game logic for Gomoku, including the board representation, game state, and rules for making moves. It provides the foundational classes and methods used by other components.

**utils.py:** Provides utility functions and helpers that support game setup, execution, and analysis.

**players.py:** Contains implementations of different player types, including human player, AI players using the Alpha-Beta pruning algorithm, and random players. **Note the human player implementation will provide a list of all legal moves at each play as asked in the instructions.**

**play_gomoku.py:** A script that sets up and runs a Gomoku game instance. Use this script to play Gomoku using the command line.

**play_analysis.ipynb:** A Jupyter notebook for analyzing different depths and evaluation functions as requested by the lab intrunctions. It includes code for running simulations, collecting gameplay metrics (like win rates and decision times), and visualizing results. 

## Playing the Game
Run the **play_gomoku.py** script to start a game. You can modify this script to set up specific player configurations (e.g. different depths), or setting two AI players to play against each other.

## Acknowledgments
- [GitHub Pages] https://github.com/aimacode 
