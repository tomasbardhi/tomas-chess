import torch
import chess
from printUtils import print_channels
from NNUtils import state_to_input
from NNUtils import decode_policy_output
from NNModel import model
from utils import find_highest_probability_legal_move

# create input for neural network from fen string
fen = 'rnbqkbnr/pppp1p1p/8/4p3/4P1pP/6P1/PPPPKP2/RNBQ1BNR b kq h3 0 1'
print("Current Board position:")
print(chess.Board(fen))
print()
board_input = state_to_input(fen)
print_channels(board_input[0])

# feed board_input to neural network and check the result
board_input_tensor = torch.tensor(board_input)
model.eval()
with torch.no_grad():
    policy, value = model(board_input_tensor)


print("Policy:", policy)
print("Value:", value)

##############################################################################################################

# convert the policy tensor to a np array and reshape it to match the 8x8x73 format
policy_np = policy.numpy().reshape(8, 8, 73)

decoded_moves = decode_policy_output(policy_np)

print("Decoded Moves:", decoded_moves)
print("Value:", value.item())


####

board = chess.Board(fen)
move = find_highest_probability_legal_move(board, decoded_moves)
print(f"Best move: {move}")
print(board)
board.push_uci(move)
print("--")
print(board)