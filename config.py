import math
import logging

logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger(__name__)

'''
    - game: logs game information
    - moves: logs each move of a game
'''
def print_file(filename, text):
    if filename == "game" or filename == "moves":
        with open("outputs/"+filename+".txt", 'a') as file:
            if not isinstance(text, str):
                text = str(text)
            file.write(text + '\n')

args = {
    'C': 2,
    'num_searches': 100,
    'num_iterations': 2,
    'num_selfPlay_iterations': 2,
    'num_epochs': 2
}

input_channels = 19 # number of channels
num_res_blocks = 10 # number of residual blocks
num_actions = 4672 # maximum possible moves