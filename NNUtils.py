import chess
import numpy as np
from NNModel import input_channels

def state_to_input(fen: str):
    board = chess.Board(fen)
    input_shape = (input_channels, 8, 8)
    state_input = np.zeros(input_shape, dtype=np.float32)
    
    # Channel indices
    index_offset = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    
    # Pieces (channels 0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = index_offset[piece.symbol()]
            rank = 7 - square // 8
            file = square % 8
            state_input[channel, rank, file] = 1    

    # Turn (channel 12)
    state_input[12, :, :] = 1 if board.turn == chess.WHITE else 0
    
    # Castling rights (channels 13-16)
    state_input[13, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    state_input[14, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    state_input[15, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    state_input[16, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0

    # Fifty-move rule (channel 17)
    state_input[17, :, :] = 1 if board.halfmove_clock >= 50 else 0

    # En passant target (channel 18)
    if board.ep_square is not None:
        ep_rank = 7 - board.ep_square // 8
        ep_file = board.ep_square % 8
        state_input[18, ep_rank, ep_file] = 1
    
    # Add batch dimension
    return state_input.reshape((1, *input_shape))