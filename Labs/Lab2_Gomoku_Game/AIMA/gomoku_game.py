from game import * 
from players import * 

def main():
    gomoku = Gomoku()
    depth = 1  # depth for the alpha-beta player
    
    # Ask the user for their preferred color
    player_color = input("Choose your color (B for Black, W for White): ").strip().upper()
    while player_color not in ['B', 'W']:
        print("Invalid color. Please choose B for Black or W for White.")
        player_color = input("Choose your color (B for Black, W for White): ").strip().upper()
    
    if player_color == 'W':
        user_player = 'W'
        ai_player = 'B'
    else:
        user_player = 'B'
        ai_player = 'W'
    
    # Initialize players
    if user_player == 'W':
        player1 = human_player  
        player2 = alpha_beta_player(depth=depth, evaluation_func=evaluate_game_state_improved) 
    else:
        player1 = alpha_beta_player(depth=depth, evaluation_func=evaluate_game_state_improved)  
        player2 = human_player 
    
    # Start the game
    winner = gomoku.play_game(player1, player2)
    
    # Determine the winner
    if (winner == 1 and user_player == 'W') or (winner == -1 and user_player == 'B'):
        print("Congratulations, you win!")
    elif winner == 0:
        print("It's a draw!")
    else:
        print("AI wins. Better luck next time!")

if __name__ == '__main__':
    main()
