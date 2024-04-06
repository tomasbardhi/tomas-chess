import chess
from NNUtils import state_to_input
from config import logger, print_file
import os
import numpy as np
from config import args
import torch

class Game:

    def __init__(self, board, mcts, model):
        self.board = board
        self.mcts = mcts
        self.model = model

    def learn(self):
        print_file("START LEARNING:")
        for iteration in range(args['num_iterations']):
            memory = []
            print_file("iteration: " + str(iteration))
            print_file(self.board)
            #print_file("memory: ")
            #print_file(memory)
            self.model.eval()
            for selfPlay_iteration in range(args['num_selfPlay_iterations']):
                print_file("self play iteration: " + str(selfPlay_iteration))
                memory += self.self_play()
                #print_file("memory:")
                #print_file(memory)
                
            self.model.train()
            #print_file("memory after train?:")
            #print_file(memory)
            '''
            for epoch in range(args['num_epochs']):
                print("epoch: " + str(epoch))
                print("memory before epoch:")
                print(memory)
                self.train(memory)
                print("memory after epoch:")
                print(memory)
            '''
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            #torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

    def self_play(self):
        storage = []

        while not self.board.is_game_over():
            # logs
            print_file("game", "----------------------------------------------------------------------------------------------------")
            print_file("game", "")
            print_file("game", "Game state:")
            print_file("game", self.board)
            print_file("game", "")
            print_file("game", "Player Turn:")
            if self.board.turn:
                print_file("game", "It's White's turn to move.")
            else:
                print_file("game", "It's Black's turn to move.")
            print_file("game", "")
            print_file("game", "Looking for a move...")
            print_file("game", "")

            # get filtered + normalized policy
            mcts_probs = self.mcts.search()
            print_file("game", "Filtered, normalized policy")
            print_file("game", mcts_probs)
            print_file("game", "")

            # save data to storage
            storage.append((self.board.copy(), mcts_probs, self.board.turn))

            # select random move from mcts_probs 
            moves = [move for move, _ in mcts_probs]
            probabilities = [prob for _, prob in mcts_probs]
            move = np.random.choice(moves, p=probabilities)

            # play move
            print_file("game", "Selected move: " + str(move))
            print_file("moves", str(move))
            self.board.push(chess.Move.from_uci(move))
            print_file("game", "")
        
        # logs
        print_file("game", "")
        print_file("game", "----------------------------------------------------------------------------------------------------")
        print_file("game", "")
        print_file("game", "Game state:")
        print_file("game", self.board)
        print_file("game", "")
        print_file("game", "Game over - " + str(self.board.result()))
        print_file("game", "")
        if self.board.is_checkmate():
            winner = 'White' if self.board.turn == chess.BLACK else 'Black'
            print_file("game", "Checkmate! " + str(winner) + " wins.")
        elif self.board.is_stalemate():
            print_file("game", "Draw! Stalemate")
        elif self.board.is_insufficient_material():
            print_file("game", "Draw! Insufficient material")
        elif self.board.is_seventyfive_moves():
            print_file("game", "Draw! 75 moves rule")
        elif self.board.is_fivefold_repetition():
            print_file("game", "Draw! Fivefold repetition.")
        elif self.board.can_claim_draw():
            print_file("game", "Draw! A draw can be claimed.")
        else:
            print_file("game", "The game ended for an unknown reason.")
        print_file("game", "")

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
            
            #print_file("game", _board_state)
            #print_file("game", "")
            #print_file("game", _mcts_probs)
            #print_file("game", "")
            #print_file("game", outcome)
            #print_file("game", "")
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