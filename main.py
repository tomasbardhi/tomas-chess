from MCTS import MCTS
import chess
from NNModel import NNModel
from Game import Game
from Trainer import Trainer
from Trainer2 import Trainer2
from config import input_channels, num_actions, num_res_blocks, args
from utils import print_file, clear_outputs
import torch
from NNUtils import state_to_input, decode_policy_output
from utils import filter_and_normalize_policy
import numpy as np
from printUtils import print_channels


def test_model():
    model_path = 'training/model_0.pt' 
    model = NNModel(input_channels, num_res_blocks, num_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    board = chess.Board(fen='7k/4Q3/5K2/8/8/8/8/8 w - - 0 1')


    #while not board.is_game_over():
    print("Board:")
    print(board)
    print("game", "")
    state_to_input__ = state_to_input(board)
    state_to_input__ = state_to_input__.reshape(input_channels, 8, 8)
    print_channels(state_to_input__)
    policy, value = model(torch.tensor(state_to_input(board)))
    policy_np = policy.detach().numpy().reshape(8, 8, 73)
    decoded_policy = decode_policy_output(policy_np)
    filtered_normalized_policy = filter_and_normalize_policy(board, decoded_policy)

    value = value.item()

    # logs
    filtered_normalized_policy_sorted = sorted(filtered_normalized_policy, key=lambda x: x[1], reverse=True)
    print("")
    print("")
    print("\nFiltered Normalized Policy:")
    print("-" * 37)
    print("{:<10} | {:>10}".format("Move", "Probability"))
    print("-" * 37)
    for move, prob in filtered_normalized_policy_sorted:
        print("{:<10} | {:>10.6f}".format(move, prob))
    print("-" * 37)
    print("")
    print("")
    print(value)
    print("")
    # logs


    best_move = max(filtered_normalized_policy, key=lambda x: x[1])[0]
    board.push(chess.Move.from_uci(best_move))

    print("")
    print("Board:")
    print(board)
    print("")

# clear output files
clear_outputs()

model = NNModel(input_channels, num_res_blocks, num_actions)
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

fen = '7k/4Q3/5K2/8/8/8/8/8 w - - 0 1'
# fen = 'r3r1k1/pbpn2b1/1p3qQ1/3p2N1/3P4/2N1P3/PP3PP1/R3K2R w - - 0 1'
#fen = '8/5k2/8/4QK2/8/8/8/8 b - - 0 1'
board = chess.Board(fen)
mcts = MCTS(board, args, model)

print_file("game", "FEN: " + fen)
print_file("game", "")
print_file("game", "Board: ")
print_file("game", board)
print_file("game", "")

game = Game(board, mcts, model)
#storage = game.self_play()

trainer = Trainer(optimizer, fen)
#trainer.learn()
#trainer.train_single_position(board)

#test_model()

### ------------------- ------------------------------------------------------------------------------------------------------------------

fen = '7k/4Q3/5K2/8/8/8/8/8 w - - 0 1'  # e7g7 is checkmate for white
model = NNModel(input_channels, num_res_blocks, num_actions)
#model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

trainer = Trainer2(model, optimizer, scheduler, fen)

policy_target = torch.zeros(1, 4672)
policy_target[0, 3825] = 1
value_target = torch.tensor([1.0])

trainer.train_position(torch.tensor(state_to_input(board)), policy_target, value_target)

trainer.model.eval()
predicted_policy, predicted_value = trainer.model(torch.tensor(state_to_input(board)))
print(f"Predicted Policy: {predicted_policy}")
print(f"Predicted Value: {predicted_value}")


### ------ train on one position ------------------------------------------------------

'''

def train_single_position(board, model, optimizer, scheduler, state_input, policy_target, value_target):
    model.train()
    print_channels(state_to_input(board).reshape(input_channels, 8, 8))
    for epoch in range(100):
        optimizer.zero_grad()
        predicted_policy, predicted_value = model(state_input)
        
        policy_loss = torch.nn.CrossEntropyLoss()(predicted_policy, policy_target)
        value_loss = torch.nn.MSELoss()(predicted_value.squeeze(0), value_target)
        total_loss = policy_loss + 0.5 * value_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}: Loss {total_loss.item()}")

        if epoch % 10 == 0:  # Print every 5 epochs to monitor predictions
            print("Predicted Policy Sample:", predicted_policy[0])
            print("Predicted Value Sample:", predicted_value.squeeze().item())
            print("Gradients of the first layer:", model.conv_base[0].weight.grad.norm().item())

            policy_np = predicted_policy[0].detach().numpy().reshape(8, 8, 73)
            decoded_policy = decode_policy_output(policy_np)
            # filtered_normalized_policy contains all the normalized legal moves ordered by probability from the actual policy
            filtered_normalized_policy = filter_and_normalize_policy(board, decoded_policy)

        # Break early if the model has effectively learned this position
        if total_loss.item() < 0.01:
            break

    torch.save(model.state_dict(), f"training/overfitted_model.pt")
    torch.save(optimizer.state_dict(), f"training/overfitted_optimizer.pt")

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


fen = '7k/4Q3/5K2/8/8/8/8/8 w - - 0 1'  # e7g7 is checkmate for white

model = NNModel(input_channels, num_res_blocks, num_actions)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

board = chess.Board(fen)
print(board)


trainer = Trainer(optimizer, fen)

random_tensor = torch.rand((1, 4672)) * 0.6
random_tensor[0, 3825] = 1
tensor = torch.zeros(1, 4672)
tensor[0, 3825] = 1
policy_target = torch.tensor([3825])    # e7g7 white checkmate
value_target = torch.tensor([1.0])

train_single_position(board, model, optimizer, scheduler, torch.tensor(state_to_input(board)), random_tensor, value_target)

model.eval()
predicted_policy, predicted_value = model(torch.tensor(state_to_input(board)))
print(f"Predicted Policy: {predicted_policy}")
print(f"Predicted Value: {predicted_value}")

'''