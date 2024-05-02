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
from NNUtils import decode_policy_output, state_to_input
from utils import filter_and_normalize_policy
from printUtils import print_channels

class Trainer2:

    def __init__(self, model, optimizer, scheduler, fen=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.board = chess.Board(fen)
        self.fen = fen

    def train_position(self, state_input, policy_target, value_target):
        self.model.train()
        print_channels(state_to_input(self.board).reshape(input_channels, 8, 8))
        for epoch in range(100):
            self.optimizer.zero_grad()
            predicted_policy, predicted_value = self.model(state_input)
            
            policy_loss = torch.nn.CrossEntropyLoss()(predicted_policy, policy_target)
            value_loss = torch.nn.MSELoss()(predicted_value.squeeze(0), value_target)
            total_loss = policy_loss + 0.5 * value_loss
            
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            print(f"Epoch {epoch}: Loss {total_loss.item()}")

            if epoch % 10 == 0:  # Print every 5 epochs to monitor predictions
                print("Predicted Policy Sample:", predicted_policy[0])
                print("Predicted Value Sample:", predicted_value.squeeze().item())
                print("Gradients of the first layer:", self.model.conv_base[0].weight.grad.norm().item())

                policy_np = predicted_policy[0].detach().numpy().reshape(8, 8, 73)
                decoded_policy = decode_policy_output(policy_np)
                filtered_normalized_policy = filter_and_normalize_policy(self.board, decoded_policy)
                print(filtered_normalized_policy)

            # Break early if the model has effectively learned this position
            if total_loss.item() < 0.01:
                break

        torch.save(self.model.state_dict(), f"training/overfitted_model.pt")
        torch.save(self.optimizer.state_dict(), f"training/overfitted_optimizer.pt")