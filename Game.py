import chess
from config import logger

class Game:

    def __init__(self, board, mcts):
        self.board = board
        self.mcts = mcts

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
        mcts_probs = self.mcts.search()
        best_move = max(mcts_probs, key=mcts_probs.get)
        logger.info("\nMCTS selected move: " + str(best_move) + "\n")
        self.board.push(chess.Move.from_uci(best_move))