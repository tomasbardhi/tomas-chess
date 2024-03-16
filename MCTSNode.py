import numpy as np
import math
import chess
from config import logger

class MCTSNode:

    def __init__(self, board, args, parent=None, move=None, prior=0):
        self.board = board

        self.args = args

        self.parent = parent
        self.move = move

        self.children = []
        
        self.prior = prior

        # not needed because we expand in all directions immediately
        #self.untried_moves = list(board.legal_moves)

        self.visits = 0
        self.wins = 0

    # a node is fully expanded when it has children (when we expand once, we expand in every direction so its either 0 or all children)
    def is_fully_expanded(self):
        return self.has_children()

    # check if node has children
    def has_children(self):
        return len(self.children) > 0

    # a node is terminal when the game has concluded
    def is_terminal_node(self):
        return self.board.is_game_over()
    
    # method to select child
    # calculate uct for every child and select child with best uct score
    def select(self):
        logger.debug("\n\tSELECT:\n")
        best_child = None
        best_uct = -np.inf
        logger.debug("\tchildren: ")  
        for child in self.children:
            uct = self.get_uct(child)
            logger.debug("\t Child "+str(child.move)+" has uct: " + str(uct) + " and probability: " + str(child.prior))
            if uct > best_uct:
                best_child = child
                best_uct = uct
        
        logger.debug("\n\tBest child:")
        logger.debug("\t Child "+str(best_child.move)+" has uct: " + str(best_uct)) 
        return best_child
    
    # create child nodes from policy moves
    def expand(self, policy):
        #logs
        logger.debug("\n\tEXPAND:\n")
        logger.debug(f"\tExpandable moves({len(policy)}):")
        moves_str = ", ".join(f"{move}" for move, _ in policy)
        logger.debug("\t" + moves_str)

        for move_uci, prob in policy:
            if prob > 0:
                # create Move object
                move = chess.Move.from_uci(move_uci)
                # copy board and make the move
                new_board = self.board.copy()
                new_board.push(move)

                # create new child node with new board, move and parent
                child = MCTSNode(new_board, self.args, self, move, prob)

                # add new child to parents children
                self.children.append(child)

    '''
    # simulate a whole game by playing random moves
    def simulate(self):
        logger.debug("\n\tSIMULATE:\n")
        # simulate on a temporary board
        temp_board = self.board.copy()
        # play until game is not over
        while not temp_board.is_game_over():
            move = np.random.choice(list(temp_board.legal_moves))
            temp_board.push(move)
        
        logger.debug(temp_board)
        # if game ends by checkmate return 1 or 0 depending on who won
        if temp_board.is_checkmate():
            if temp_board.turn == self.board.turn:
                logger.debug("White loses")
                return 1
            else:
                logger.debug("Black loses")
                return 0
        # if game ends for any other reason (draw), return 0.5
        elif temp_board.is_game_over():
            return 0.5
        # if game ends for any other reason, return 0.5
        return 0.5
    '''

    # update all wins and visits of each parent node starting from the last node
    def backpropagate(self, result):
        self.wins += result
        self.visits += 1
        if self.parent is not None:
            self.parent.backpropagate(result)

    '''
    # ucb is used to find the best next node
    def get_ucb(self, child):
        if child.visits == 0:
            return float('inf')

        winrate = child.wins / child.visits
        exploration = self.args['C'] * math.sqrt(math.log(self.visits) / child.visits)
        return winrate + exploration
    '''
        
    # revisited ucb for alphazero -> upper confidence bound applied to trees
    def get_uct(self, child):
        q = child.wins / child.visits if child.visits > 0 else 0
        # total visits of all siblings + current child
        total_visits = sum(parent.visits for parent in self.children)
        uct = q + self.args['C'] * child.prior * math.sqrt(total_visits) / (1 + child.visits)
        return uct