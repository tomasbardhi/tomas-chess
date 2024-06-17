import math
import logging

logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger(__name__)

args = {
    'C': 2,
    'num_searches': 500,
    'num_iterations': 100,
    'num_selfPlay_iterations': 100,
    'num_epochs': 5,
    'batch_size': 32
}

input_channels = 19 # number of channels
num_res_blocks = 10 # number of residual blocks
num_actions = 4672 # maximum possible moves