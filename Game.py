import chess
from NNUtils import state_to_input
from config import logger
import os
import numpy as np

class Game:

    def __init__(self, board, mcts, model):
        self.board = board
        self.mcts = mcts
        self.model = model

    def self_play(self):
        storage = []

        while not self.board.is_game_over():
            # logs
            os.system('cls')
            print("\nGame state:")
            logger.error(self.board)
            print("\nPlayer Turn:")
            if self.board.turn:
                print("It's White's turn to move.")
            else:
                print("It's Black's turn to move.")

            # get filtered + normalized policy
            mcts_probs = self.mcts.search()
            print("\nFiltered, normalized policy:")
            print(mcts_probs)

            # save data to storage
            storage.append((self.board.copy(), mcts_probs, self.board.turn))

            # select random move from mcts_probs 
            moves = [move for move, _ in mcts_probs]
            probabilities = [prob for _, prob in mcts_probs]
            move = np.random.choice(moves, p=probabilities)

            # play move
            logger.info("\nSelected move: " + str(move) + "\n")
            self.board.push(chess.Move.from_uci(move))
        
        # logs
        logger.error("Game over.")
        print(self.board.result())

        # 1 if white wins, -1 if black wins, 0 if draw
        winner = 0
        if self.board.is_checkmate(): 
            winner = 1 if self.board.turn == chess.BLACK else -1

        processed_storage = []
        for _board_state, _mcts_probs, _player_turn in storage:
            # transform board to input
            state_input = state_to_input(_board_state)
            # set outcome
            if winner == 0:
                outcome = 0
            else:
                outcome = 1 if (winner == 1 and _player_turn == chess.WHITE) or (winner == -1 and _player_turn == chess.BLACK) else -1
            
            print(_board_state)
            print(_mcts_probs)
            print(outcome)
            print("\n\n")
            processed_storage.append((state_input, _mcts_probs, outcome))

        return processed_storage

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
        print("MCTS SEARCH RETURN VALUES:")
        print(mcts_probs)
        best_move = max(mcts_probs, key=lambda x: x[1])[0]
        logger.info("\nMCTS selected move: " + str(best_move) + "\n")
        self.board.push(chess.Move.from_uci(best_move))