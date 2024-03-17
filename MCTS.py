from MCTSNode import MCTSNode
import logging
from NNUtils import decode_policy_output, state_to_input
from utils import filter_and_normalize_policy
import torch
from config import logger

class MCTS:

    def __init__(self, board, args, model):
        self.board = board
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self):
        # initialize root node
        root = MCTSNode(self.board, self.args)
        logger.debug("\nSEARCH:")
        filtered_normalized_policy = []
        for _ in range(self.args['num_searches']):
            node = root
            logger.debug("\nIteration: " + str(_))  
            # selection
            #print(node.children)
            selection_depth = 0
            while node.is_fully_expanded():
                logger.info("SELECTION DEPTH: " + str(selection_depth))
                node = node.select()
                selection_depth = selection_depth + 1

            if not node.is_terminal_node():
                # feed board_input to neural network and get the results (policy, value)
                policy, value = self.model(torch.tensor(state_to_input(node.board)))

                # convert the policy tensor to a np array and reshape it to match the 8x8x73 format
                policy_np = policy.numpy().reshape(8, 8, 73)
                decoded_policy = decode_policy_output(policy_np)
                # filtered_normalized_policy contains all the normalized legal moves ordered by probability from the actual policy
                filtered_normalized_policy = filter_and_normalize_policy(node.board, decoded_policy)

                # value contains the value returned by the nn for the given board
                value = value.item()
                
                # expansion
                node.expand(filtered_normalized_policy)  
            
            # backpropagation
            node.backpropagate(value)
        

        # return moves probabilities
        logger.info("\nPROBS:\n")

        moves_probs = {}
        move_stats = {}
        total_visits = 0

        logger.info(f"{'Move':<10} | {'Visits':<10} | {'Probability':<15}")
        logger.info("-" * 45)

        # calc move visits and total vists
        for child in root.children:
            total_visits += child.visits
            move_stats[child.move.uci()] = {'visits': child.visits}

        # calc probabilities
        for move in move_stats:
            move_stats[move]['probability'] = move_stats[move]['visits'] / total_visits

        # print
        for move, stats in move_stats.items():
            logger.info(f"{move:<10} | {stats['visits']:<10} | {stats['probability']:<15}")

        logger.info("\nTotal visits: " + str(total_visits))

        #moves_probs = {move: stats['probability'] for move, stats in move_stats.items()}
        return [(move, stats['probability']) for move, stats in move_stats.items()]