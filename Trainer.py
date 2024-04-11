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
                self.train_02(memory)
            
            
            print_file("self-learn", "")
            print_file("self-learn", "--------------------------------------------------------------------------------")
            print_file("self-learn", "")  

            torch.save(self.game.model.state_dict(), f"training/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"training/optimizer_{iteration}.pt")

    def train_02(self, memory):
        #random.shuffle(memory)
        for batchIdx in range(0, len(memory), args['batch_size']):
            sample = memory[batchIdx:batchIdx + args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)
            
            # pad policy
            policy_targets = [np.pad(policy, (0, 218 - len(policy)), 'constant') for policy in policy_targets]


            print(state)
            print(policy_targets)
            print(value_targets)
            
            print("TEST: TEST: TEST: TEST: TEST: TEST: TEST: TEST: TEST: TEST: TEST: ")
            print("")
            state = torch.tensor(state, dtype=torch.float32)
            state = state.squeeze(1)
            print(state)
            print("")
            print("")
            print(policy_targets)
            print("")
            print("")
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            print(policy_targets)
            print("")
            value_targets = torch.tensor(value_targets, dtype=torch.float32).view(-1, 1)
            print(value_targets)
            print("")
            
            out_policy, out_value = self.game.model(state)

            print("OUT POLICY")
            print("")
            print(out_policy)
            print("")
            print("")
            print(policy_targets)
            print("")
            print("")
            print("")
            print("")
            print("")
            print("")

            for s in state:
                print("State: ")
                print("")
                print_channels(s)
                print("")
                print("")
                print("")

            #policy_np = out_policy.detach().numpy().reshape(8, 8, 73)
            #print(out_policy.shape)
            #decoded_policy = decode_policy_output(policy_np)
            

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_01(self, memory):
        # create tensors
        print(memory[0][0])
        print(memory.shape)
        state_inputs = torch.stack([torch.tensor(m[0], dtype=torch.float).squeeze(0) for m in memory])
        policies = torch.stack([torch.tensor(m[1], dtype=torch.float) for m in memory])
        values = torch.tensor([m[2] for m in memory], dtype=torch.float)

        print(state_inputs.shape)
        print(state_inputs[0].shape)
        print(policies.shape)
        print(policies[0].shape)
        print(values.shape)
        print(values[0].shape)

        # dataset + loader for batching
        train_dataset = TensorDataset(state_inputs, policies, values)
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

        # loss functions
        policy_loss_fn = torch.nn.CrossEntropyLoss()
        value_loss_fn = torch.nn.MSELoss()

        # loop through batches
        for state_batch, policy_batch, value_batch in train_loader:
            # model prediction^
            predicted_policy, predicted_value = self.game.model(state_batch)

            # loss calculation
            print("predicted_policy.shape")
            print(predicted_policy.shape)
            print(predicted_policy.squeeze().shape)
            print("predicted_value.shape")
            print(predicted_value.shape)
            print(predicted_value.squeeze().shape)
            policy_loss = policy_loss_fn(predicted_policy, policy_batch)
            print("fine")
            value_loss = value_loss_fn(predicted_value.squeeze(), value_batch)
            total_loss = policy_loss + value_loss

            # backpropagation
            self.optimizer.zero_grad() 
            total_loss.backward()
            self.optimizer.step()
