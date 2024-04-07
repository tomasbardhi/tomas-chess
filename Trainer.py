import chess
from utils import print_file
from config import args
import torch
from Game import Game
from MCTS import MCTS
from NNModel import NNModel
from config import input_channels, num_actions, num_res_blocks, args

class Trainer:

    def __init__(self, fen=None):
        self.fen = fen
        model = NNModel(input_channels, num_res_blocks, num_actions)
        self.board = self.create_board()
        self.mcts = MCTS(self.board, args, model)
        self.game = Game(self.board, self.mcts, model)

    def create_board(self):
        if self.fen:
            try:
                return chess.Board(self.fen)
            except ValueError:
                print("FEN error. Creating standard board instead!")
        return chess.Board()

    def reset(self):
        self.board = self.create_board()
        self.mcts.board = self.board
        self.game.board = self.board

    def learn(self):
        print_file("self-learn", "Start of self learning")
        print_file("self-learn", "")

        for iteration in range(args['num_iterations']):
            
            # reset board & memory
            memory = []

            print_file("self-learn", "Iteration: " + str(iteration))
            print_file("self-learn", "")
            #print_file("memory: ")
            #print_file(memory)
            self.game.model.eval()
            for selfPlay_iteration in range(args['num_selfPlay_iterations']):
                # reset boards
                self.reset()
                print_file("self-learn", "--------------------------------------------------------------------------------")
                print_file("self-learn", "")
                print_file("self-learn", "Game: " + str(selfPlay_iteration))
                print_file("self-learn", "")
                print_file("self-learn", "Board start: ")
                print_file("self-learn", "")
                print_file("self-learn", self.game.board)
                print_file("self-learn", "")
                memory += self.game.self_play()
                #print_file("memory:")
                #print_file(memory)
                        
            
            print_file("self-learn", "")
            print_file("self-learn", "--------------------------------------------------------------------------------")
            print_file("self-learn", "")         

            '''    
            self.game.model.train()
            for epoch in range(args['num_epochs']):
                print("epoch: " + str(epoch))
                print("memory before epoch:")
                print(memory)
                self.train(memory)
                print("memory after epoch:")
                print(memory)
            '''
            #torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            #torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
