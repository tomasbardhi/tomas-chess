import chess
import numpy as np

def find_highest_probability_legal_move(board, policy_output):
    legal_moves = {board.uci(move) for move in board.legal_moves}
    all_moves = policy_output

    legal_probable_moves = [(move, probability) for move, probability in policy_output if move in legal_moves]

    print("Top 10 legal moves based on probability:")
    for i, (move, probability) in enumerate(legal_probable_moves[:10], start=1):
        print(f"{i}. Move: {move}, Probability: {probability}")

    if legal_probable_moves:
        return legal_probable_moves[0][0]
    else:
        return None