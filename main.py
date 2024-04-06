from MCTS import MCTS
import chess
from NNModel import NNModel
from Game import Game
from config import input_channels, num_actions, num_res_blocks, args, print_file
import os

# clear all output files 
for filename in os.listdir("outputs"):
    file_path = os.path.join("outputs", filename)
    if os.path.isfile(file_path):
        with open(file_path, 'w'):
            pass

model = NNModel(input_channels, num_res_blocks, num_actions)
model.eval()

fen = '7k/4Q3/5K2/8/8/8/8/8 b - - 0 1'
board = chess.Board(fen)
mcts = MCTS(board, args, model)

print_file("game", "FEN: " + fen)
print_file("game", "")
print_file("game", "Board: ")
print_file("game", board)
print_file("game", "")

game = Game(board, mcts, model)
storage = game.self_play()
#game.learn()