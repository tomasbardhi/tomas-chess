from MCTS import MCTS
import chess
import numpy as np
import math
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

args = {
    'C': math.sqrt(2),
    'num_searches': 1000
}

class Game:

    def __init__(self):
        self.board = chess.Board()

    def play_game(self):
        logger.info("\nPick a legal move from the list and type it to play.")

        while not self.board.is_game_over():
            logger.info("\nCurrent board position:")
            logger.info(self.board)
            
            if self.board.turn == chess.WHITE:
                self.player_move()
            else:
                logger.info("MCTS is thinking...")
                self.mcts_move()
        
        logger.info("Game over.")
        logger.info("Result:", self.board.result())

    def player_move(self):
        legal_moves = [move.uci() for move in self.board.legal_moves]
        logger.info("\nLegal moves:\n")
        logger.info(legal_moves)
        move = input("\nYour move: ")
        while move not in legal_moves:
            logger.info("\nIllegal move. Please try again.")
            move = input("\nYour move: ")
        self.board.push_uci(move)

    def mcts_move(self):
        mcts = MCTS(self.board.copy(), args)
        mcts_probs = mcts.search()
        best_move = max(mcts_probs, key=mcts_probs.get)
        logger.info("\nMCTS selected move: " + str(best_move) + "\n")
        self.board.push(chess.Move.from_uci(best_move))

if __name__ == "__main__":
    game = Game()
    game.play_game()