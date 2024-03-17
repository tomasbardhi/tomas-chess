from MCTS import MCTS
import chess
from NNModel import NNModel
from Game import Game
from config import input_channels, num_actions, num_res_blocks, args

model = NNModel(input_channels, num_res_blocks, num_actions)
model.eval()

fen = '7k/4Q3/5K2/8/8/8/8/8 b - - 0 1'
board = chess.Board(fen)
print(board)
mcts = MCTS(board, args, model)

game = Game(board, mcts, model)
storage = game.self_play()