from game import * 
from players import * 

def main():
    gomoku = Gomoku()
    depth = 3  # depth for the alpha-beta player
    
    # Ask the user for their preferred color
    player_color = input("Choose your color (B for Black, W for White): ").strip().upper()
    while player_color not in ['B', 'W']:
        print("Invalid color. Please choose B for Black or W for White.")
        player_color = input("Choose your color (B for Black, W for White): ").strip().upper()
    
    # Assign players based on the user's choice
    if player_color == 'W':
        player1, player2 = human_player, alpha_beta_player(depth=depth, evaluation_func=evaluate_game_state_improved)
    else:
        player1, player2 = alpha_beta_player(depth=depth, evaluation_func=evaluate_game_state_improved), human_player

    # Start the game
    winner = gomoku.play_game(player1, player2)

    # Determine the winner based on the game's utility value and the user's chosen color
    if winner == 1:  # 'B' wins
        print("Congratulations, you win!" if player_color == 'B' else "AI wins. Better luck next time!")
    elif winner == -1:  # 'W' wins
        print("AI wins. Better luck next time!" if player_color == 'B' else "Congratulations, you win!")
    else:
        print("It's a draw!")


if __name__ == '__main__':
    main()
