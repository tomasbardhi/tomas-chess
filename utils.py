import chess
import numpy as np
import os



def clear_outputs():
    for filename in os.listdir("outputs"):
        file_path = os.path.join("outputs", filename)
        if os.path.isfile(file_path):
            with open(file_path, 'w'):
                pass

'''
    - game: logs game information
    - moves: logs each move of a game
'''
def print_file(filename, text):
    if filename == "ggame" or filename == "mmoves" or filename == "self-learn":
        with open("outputs/"+filename+".txt", 'a') as file:
            if not isinstance(text, str):
                text = str(text)
            file.write(text + '\n')

def normalize_policy(policy):
    # probabilities of policy legal moves
    probabilities = [prob for _, prob in policy]

    # normalization
    sum_probabilities = sum(probabilities)
    normalized_probabilities = [prob / sum_probabilities for prob in probabilities]

    # new normalized policy
    normalized_policy = list(zip([move for move, _ in policy], normalized_probabilities))
    return normalized_policy

def filter_policy(board, policy):
    legal_moves = [move.uci() for move in board.legal_moves]

    # keep only legal moves
    filtered_policy = [(move, prob) for move, prob in policy if move in legal_moves]
    
    return filtered_policy

def filter_and_normalize_policy(board, policy, epsilon=1e-5):
    legal_moves = [move.uci() for move in board.legal_moves]

    # keep only legal moves
    filtered_policy = [(move, prob) for move, prob in policy if move in legal_moves]

    # probabilities of policy legal moves
    probabilities = [prob for _, prob in filtered_policy]

    # normalization
    sum_probabilities = sum(probabilities)
    if(sum_probabilities == 0):
        normalized_probabilities = [epsilon for prob in probabilities]
    else:
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