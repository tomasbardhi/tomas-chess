import chess
import numpy as np

def filter_and_normalize_policy(board, policy):
    legal_moves = [move.uci() for move in board.legal_moves]

    # keep only legal moves
    filtered_policy = [(move, prob) for move, prob in policy if move in legal_moves]

    # probabilities of policy legal moves
    probabilities = [prob for _, prob in filtered_policy]

    # normalization
    sum_probabilities = sum(probabilities)
    normalized_probabilities = [prob / sum_probabilities for prob in probabilities]

    # new filtered normalized policy
    normalized_policy = list(zip([move for move, _ in filtered_policy], normalized_probabilities))
    return normalized_policy

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