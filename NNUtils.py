import chess
import numpy as np
from config import input_channels

def state_to_input(board):
    input_shape = (input_channels, 8, 8)
    state_input = np.zeros(input_shape, dtype=np.float32)
    
    # channel indices for white/black pieces
    index_offset = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # white pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # black pieces
    }
    
    # pieces (channels 0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = index_offset[piece.symbol()]
            rank = 7 - square // 8
            file = square % 8
            state_input[channel, rank, file] = 1    

    # turn (channel 12)
    state_input[12, :, :] = 1 if board.turn == chess.WHITE else 0
    
    # castling rights (channels 13-16)
    state_input[13, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    state_input[14, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    state_input[15, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    state_input[16, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0

    # fifty-move rule (channel 17)
    state_input[17, :, :] = 1 if board.halfmove_clock >= 50 else 0

    # en passant (channel 18)
    if board.ep_square is not None:
        ep_rank = 7 - board.ep_square // 8
        ep_file = board.ep_square % 8
        state_input[18, ep_rank, ep_file] = 1
    
    # add batch dimension
    return state_input.reshape((1, *input_shape))

def decode_policy_output(policy_output):
    all_moves = []  # all decoded moves (move, probability)

    for i in range(8):  # row
        for j in range(8):  # column
            for k in range(73):  # plane
                probability = policy_output[i, j, k]
                move = decode_move(i, j, k)
                if move:
                    all_moves.append((move, probability))
    
    # sorty by probability
    all_moves.sort(key=lambda x: x[1], reverse=True)

    #return [move for move, _ in all_moves]
    return all_moves


def square_to_coordinates(square):
    # square index (0-63) to notation ('a1')
    return chess.SQUARE_NAMES[square]

def decode_move(row, col, plane):
    from_square = row * 8 + col
    to_square = None
    promotion_piece = None

    if plane < 56:  # queenlike moves
        direction, distance = divmod(plane, 8)
        # 8 directions (n, ne, e, se, s, sw, w, nw)
        dir_offsets = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        offset = dir_offsets[direction]
        to_row = row + offset[0] * (distance + 1)
        to_col = col + offset[1] * (distance + 1)
        if 0 <= to_row < 8 and 0 <= to_col < 8:
            to_square = to_row * 8 + to_col
    elif plane < 64:  # knightlike moves
        knight_offsets = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2)]
        offset = knight_offsets[plane - 56]
        to_row = row + offset[0]
        to_col = col + offset[1]
        if 0 <= to_row < 8 and 0 <= to_col < 8 and abs(col - to_col) <= 2:
            to_square = to_row * 8 + to_col
    else:  # promotions (knight, bishop, rook)
        if (row == 1 and plane >= 64 and plane <= 72) or (row == 6 and plane >= 64 and plane <= 72):
            # pawn promotion based on color
            if row == 6:  # white pawn promotion
                direction_offsets = [(1, -1), (1, 0), (1, 1)]  # left, forward, right
                promotion_row = 7
            else:  # black pawn promotion
                direction_offsets = [(-1, -1), (-1, 0), (-1, 1)]  # left, forward, right
                promotion_row = 0

            underpromotion_offset_index = (plane - 64) % 3
            offset = direction_offsets[underpromotion_offset_index]
            to_row = row + offset[0]
            to_col = col + offset[1]

            if to_row == promotion_row and 0 <= to_col < 8:
                to_square = to_row * 8 + to_col
                promotion_piece_type = (plane - 64) // 3
                promotion_pieces = ['n', 'b', 'r']
                promotion_piece = promotion_pieces[promotion_piece_type]

    if to_square is not None and 0 <= to_square < 64:
        move_str = square_to_coordinates(from_square) + square_to_coordinates(to_square)
        if promotion_piece:
            move_str += promotion_piece
        return move_str
    return None

