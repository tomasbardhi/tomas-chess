from MCTS import MCTS
import chess
from NNModel import NNModel
from Game import Game
from Trainer import Trainer
from config import input_channels, num_actions, num_res_blocks, args
from utils import print_file, clear_outputs


# clear output files
clear_outputs()

model = NNModel(input_channels, num_res_blocks, num_actions)
model.eval()

#fen = '7k/4Q3/5K2/8/8/8/8/8 w - - 0 1'
fen = 'r3r1k1/pbpn2b1/1p3qQ1/3p2N1/3P4/2N1P3/PP3PP1/R3K2R w - - 0 1'
board = chess.Board()
mcts = MCTS(board, args, model)

print_file("game", "FEN: " + fen)
print_file("game", "")
print_file("game", "Board: ")
print_file("game", board)
print_file("game", "")

game = Game(board, mcts, model)
#storage = game.self_play()
trainer = Trainer(fen)
trainer.learn()