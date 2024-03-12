from MCTS import MCTS
import chess
from NNModel import NNModel
from Game import Game
from config import input_channels, num_actions, num_res_blocks, args

'''
model = NNModel(input_channels, num_res_blocks, num_actions)
model.eval()

board = chess.Board()

mcts = MCTS(board, args, model)

game = Game(board, mcts)
game.play_game()
'''

board = chess.Board()
print(board.legal_moves)