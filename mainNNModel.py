import torch
import chess
from printUtils import print_channels
from NNUtils import state_to_input
from NNUtils import decode_policy_output
from NNModel import NNModel
from utils import find_highest_probability_legal_move
from utils import filter_and_normalize_policy
from config import input_channels, num_actions, num_res_blocks, args


model = NNModel(input_channels, num_res_blocks, num_actions)
model.eval()

# create input for neural network from fen string
fen = 'rnbqkbnr/pppp1p1p/8/4p3/4P1pP/6P1/PPPPKP2/RNBQ1BNR b kq h3 0 1'
board = chess.Board(fen)
print("Current Board position:")
print(board)
print()
board_input = state_to_input(board)
print_channels(board_input[0])

# feed board_input to neural network and check the result
board_input_tensor = torch.tensor(board_input)
model.eval()
with torch.no_grad():
    policy, value = model(board_input_tensor)


print("Policy:", policy)
print("Value:", value)

##############################################################################################################


policy_np = policy.numpy().reshape(8, 8, 73)
print(policy_np)

decoded_moves = decode_policy_output(policy_np)
filtered_normalized_policy = filter_and_normalize_policy(board, decoded_moves)

print([move for move, _ in decoded_moves])
print([move.uci() for move in board.legal_moves])
print([move for move, _ in filtered_normalized_policy])
print("LEGAL MOVES:", len(([move.uci() for move in board.legal_moves])))
print("Decoded Moves:", len(filtered_normalized_policy))
print("Value:", value.item())
print(filtered_normalized_policy)

'''
move = find_highest_probability_legal_move(board, decoded_moves)
print(f"Best move: {move}")
print(board)
board.push_uci(move)
print("--")
print(board)
'''