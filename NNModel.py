import torch
import torch.nn as nn
import torch.nn.functional as F

class NNModel(nn.Module):
    # convolutional base for feature extraction + two heads for policy (move probabilities) and value (winning chances).
    def __init__(self, input_channels, num_res_blocks, num_actions):
        super(NNModel, self).__init__()
        # Convolutional Base
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Help the network learn deeper
        self.res_blocks = nn.Sequential(
            *[ResBlock(128) for _ in range(num_res_blocks)]
        )
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, num_actions),
            nn.Softmax(dim=1)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    # convolutional base -> residual blocks
    #                                       -> policy 
    #                                       -> value
    def forward(self, x):
        x = self.conv_base(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class ResBlock(nn.Module):
    # Output of the block is added to its input to help the network learn deeper features without losing important information from earlier layers.
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


input_channels = 19 # Number of channels
num_res_blocks = 10 # Number of residual blocks
num_actions = 4672 # Maximum possible moves

model = NNModel(input_channels, num_res_blocks, num_actions)