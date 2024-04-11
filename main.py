from MCTS import MCTS
import chess
from NNModel import NNModel
from Game import Game
from Trainer import Trainer
from config import input_channels, num_actions, num_res_blocks, args
from utils import print_file, clear_outputs
import torch
from NNUtils import state_to_input, decode_policy_output
from utils import filter_and_normalize_policy
import numpy as np
from printUtils import print_channels

# clear output files
clear_outputs()

model = NNModel(input_channels, num_res_blocks, num_actions)
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

fen = '7k/4Q3/5K2/8/8/8/8/8 b - - 0 1'
# fen = 'r3r1k1/pbpn2b1/1p3qQ1/3p2N1/3P4/2N1P3/PP3PP1/R3K2R w - - 0 1'
board = chess.Board()
mcts = MCTS(board, args, model)

print_file("game", "FEN: " + fen)
print_file("game", "")
print_file("game", "Board: ")
print_file("game", board)
print_file("game", "")

game = Game(board, mcts, model)
#storage = game.self_play()

trainer = Trainer(optimizer, fen)
trainer.learn()


def test_model():
    model_path = 'training/model_0.pt' 
    model = NNModel(input_channels, num_res_blocks, num_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    board = chess.Board(fen='7k/4Q3/5K2/8/8/8/8/8 w - - 0 1')


    #while not board.is_game_over():
    print("Board:")
    print(board)
    print("game", "")
    state_to_input__ = state_to_input(board)
    state_to_input__ = state_to_input__.reshape(input_channels, 8, 8)
    print_channels(state_to_input__)
    policy, value = model(torch.tensor(state_to_input(board)))
    policy_np = policy.detach().numpy().reshape(8, 8, 73)
    decoded_policy = decode_policy_output(policy_np)
    filtered_normalized_policy = filter_and_normalize_policy(board, decoded_policy)

    value = value.item()
    print("")
    print("")
    print(filtered_normalized_policy)
    print("")
    print(value)
    print("")
    board.push(chess.Move.from_uci(filtered_normalized_policy[0][0]))



    print("")
    print("Board:")
    print(board)
    print("")
