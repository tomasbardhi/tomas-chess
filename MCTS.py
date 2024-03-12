from MCTSNode import MCTSNode
import logging
from NNUtils import decode_policy_output, state_to_input
from utils import filter_and_normalize_policy
import torch

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

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
        for _ in range(self.args['num_searches']):
            node = root
            logger.debug("\nIteration: " + str(_))  
            # selection
            while node.is_fully_expanded():
                node = node.select()

            if not node.is_terminal_node():
                # feed board_input to neural network and get the results (policy, value)
                policy, value = self.model(torch.tensor(state_to_input(self.board)))

                # convert the policy tensor to a np array and reshape it to match the 8x8x73 format
                policy_np = policy.numpy().reshape(8, 8, 73)
                decoded_policy = decode_policy_output(policy_np)
                filtered_normalized_policy = filter_and_normalize_policy(decoded_policy)

                



                # expansion
                node = node.expand()    
                # simulation
                result = node.simulate()
                logger.debug("\tSimulation result: " + str(result))
            
            # backpropagation
            node.backpropagate(result)
        

        # return moves probabilities
        logger.info("\nPROBS:\n")

        moves_probs = {}
        move_stats = {}
        total_visits = 0

        logger.info(f"{'Move':<10} | {'Visits':<7} | {'Win Rate':<10} | {'Probability':<12}")
        logger.info("-" * 45)

        # calc move visits, winrate and total vists
        for child in root.children:
            total_visits += child.visits
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            move_stats[child.move.uci()] = {'visits': child.visits, 'win_rate': win_rate}

        # calc probabilities
        for move in move_stats:
            move_stats[move]['probability'] = move_stats[move]['visits'] / total_visits

        # print
        for move, stats in move_stats.items():
            logger.info(f"{move:<10} | {stats['visits']:<7} | {stats['win_rate']:<10.2%} | {stats['probability']:<12}")

        logger.info("\nTotal visits: " + str(total_visits))

        #moves_probs = {move: stats['probability'] for move, stats in move_stats.items()}
        return {move: stats['probability'] for move, stats in move_stats.items()}