import numpy as np
import math
import chess
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

class MCTSNode:

    def __init__(self, board, args, parent=None, move=None):
        self.board = board

        self.args = args

        self.parent = parent
        self.move = move

        self.children = []
        self.untried_moves = list(board.legal_moves)

        self.visits = 0
        self.wins = 0

    # a node is fully expanded when it has children and no untried moves
    def is_fully_expanded(self):
        return not self.is_expandable_node() and self.has_children()

    # check if node has children
    def has_children(self):
        return len(self.children) > 0

    # a node that has untried moves
    def is_expandable_node(self):
        return len(self.untried_moves) > 0

    # a node is terminal when the game has concluded
    def is_terminal_node(self):
        return self.board.is_game_over()

    # a node is fully expanded when all possible moves from that node have been explored. 
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    # method to select child
    # calculate ucb for every child and select child with best ucb score
    def select(self):
        logger.debug("\n\tSELECT:\n")
        best_child = None
        best_ucb = -np.inf
        logger.debug("\tchildren: ")  
        for child in self.children:
            ucb = self.get_ucb(child)
            logger.debug("\t Child "+str(child.move)+" has ucb: " + str(ucb))
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        logger.debug("\n\tBest child:")
        logger.debug("\t Child "+str(best_child.move)+" has ucb: " + str(best_ucb)) 
        return best_child
    
    # create a child node from one untried move
    def expand(self):
        logger.debug("\n\tEXPAND:\n")
        logger.debug("\tExpandable moves("+str(len(self.untried_moves))+"):")
        logger.debug("\t"),
        moves_str = ", ".join(str(move) for move in self.untried_moves)
        logger.debug("\t" + moves_str)
        # pick random untried move
        move = np.random.choice(self.untried_moves)
        logger.debug("\tPicked random move: " + str(move))
        # copy board and make the move
        new_board = self.board.copy()
        new_board.push(move)

        # create new child node with new board, move and parent
        child = MCTSNode(new_board, self.args, move=move, parent=self)
        
        # add new child to parents children
        self.children.append(child)
        
        # remove played move from parents untried moves
        self.untried_moves.remove(move)


        logger.debug("\tExpandable moves after removing move:")
        logger.debug("\t")
        moves_str = ", ".join(str(move) for move in self.untried_moves)
        logger.debug("\t" + moves_str)
        logger.debug("")

        return child

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

    # update all wins and visits of each parent node starting from the last node
    def backpropagate(self, result):
        self.wins += result
        self.visits += 1
        if self.parent is not None:
            self.parent.backpropagate(result)

    # ucb is used to find the best next node
    def get_ucb(self, child):
        if child.visits == 0:
            return float('inf')

        winrate = child.wins / child.visits
        exploration = self.args['C'] * math.sqrt(math.log(self.visits) / child.visits)
        return winrate + exploration
