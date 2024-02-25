import torch
import chess
from printUtils import print_channels
from NNUtils import state_to_input
from NNModel import model

# create input for neural network from fen string
fen = 'rnbqkbnr/pppp1p1p/8/4p3/4P1pP/6P1/PPPPKP2/RNBQ1BNR b kq h3 0 1'
board_input = state_to_input(fen)
print_channels(board_input[0])

# feed board_input to neural network and check the result
board_input_tensor = torch.tensor(board_input)
model.eval()
with torch.no_grad():
    policy, value = model(board_input_tensor)
    
print("Policy:", policy)
print("Value:", value)