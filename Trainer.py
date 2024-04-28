import chess
from utils import print_file
from config import args
import torch
from Game import Game
from MCTS import MCTS
from NNModel import NNModel
from config import input_channels, num_actions, num_res_blocks, args
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import torch.nn.functional as F
from NNUtils import decode_policy_output
from utils import filter_and_normalize_policy
from printUtils import print_channels

class Trainer:

    def __init__(self, optimizer, fen=None):
        self.optimizer = optimizer
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

                        
            
            print_file("self-learn", "")
            print_file("self-learn", "--------------------------------------------------------------------------------")
            print_file("self-learn", "")         

            
            print_file("self-learn", "")
            print_file("self-learn", "Training...")
            print_file("self-learn", "")  
            self.game.model.train()
            for epoch in range(args['num_epochs']):
                print_file("self-learn", "")
                print_file("self-learn", "--------------------------------------------------------------------------------")
                print_file("self-learn", "")  
                print_file("self-learn", "Epoch " + str(epoch))  
                self.train(memory)
            
            
            print_file("self-learn", "")
            print_file("self-learn", "--------------------------------------------------------------------------------")
            print_file("self-learn", "")  

            torch.save(self.game.model.state_dict(), f"training/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"training/optimizer_{iteration}.pt")

    def train(self, memory):
        # create tensors
        # memory[0] - state input
        # memory[1] - raw prediction
        # memory[2] - result
        state_inputs = torch.stack([torch.tensor(m[0], dtype=torch.float) for m in memory]).squeeze(1)
        policies = torch.stack([m[1][0] for m in memory]).squeeze(1)
        values = torch.tensor([m[2] for m in memory], dtype=torch.float)

        # dataset + loader for batching
        train_dataset = TensorDataset(state_inputs, policies, values)
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

        # loss functions
        policy_loss_fn = torch.nn.CrossEntropyLoss()
        value_loss_fn = torch.nn.MSELoss()

        # loop through batches
        for state_batch, policy_batch, value_batch in train_loader:
            # model prediction
            predicted_policy, predicted_value = self.game.model(state_batch)

            # loss calculation
            policy_loss = policy_loss_fn(predicted_policy, policy_batch.max(1)[1])
            value_loss = value_loss_fn(predicted_value.squeeze(), value_batch)
            total_loss = policy_loss + value_loss
            print(total_loss)

            # backpropagation
            self.optimizer.zero_grad() 
            total_loss.backward()
            #print(self.game.model.conv_base[0].weight.grad)
            self.optimizer.step()
