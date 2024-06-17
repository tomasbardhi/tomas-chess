from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import torch
from NNModel import NNModel
from MCTS import MCTS
from config import input_channels, num_res_blocks, num_actions, args
from NNUtils import state_to_input, decode_policy_output
from utils import filter_and_normalize_policy
from printUtils import print_channels
import numpy as np
from OpeningsEngine import find_opening_move

# uvicorn server:app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    fen: str

model_path = 'training/model_05062024.pt' 
model = NNModel(input_channels, num_res_blocks, num_actions)
#set model + model path to start the network with the model
#model.load_state_dict(torch.load(model_path))
model.eval()    

def count_defenders(board, square):
    piece = board.piece_at(square)
    if piece is None:
        raise ValueError("There is no piece on the given square")
    
    piece_color = piece.color

    defenders = board.attackers(piece_color, square)

    num_defenders = sum(1 for _ in defenders)
    
    return num_defenders

def count_attackers(board, square, color):
    attackers = board.attackers(color, square)
    print(attackers)
    print(" ")

    num_attackers = sum(1 for _ in attackers)
    
    return num_attackers

def is_blunder(board, move):

    temp_board = board.copy()
    temp_board.push(move)

    # check if piece not defended
    if move in board.legal_moves:
        print(move)
        print("Defenses:")
        print(count_defenders(temp_board, move.to_square))
        print("Attackers:")
        print(count_attackers(temp_board, move.to_square, chess.WHITE))
        print("------")
        if count_defenders(temp_board, move.to_square) < count_attackers(temp_board, move.to_square, chess.WHITE):
            return True
        
    # check if all pieces sufficiently defended
    for square in chess.SQUARES:
        piece = temp_board.piece_at(square)
        if piece is not None and piece.color == chess.BLACK:
            if count_defenders(temp_board, square) < count_attackers(temp_board, square, chess.WHITE):
                print(piece, square)
                print("Not sufficiently defended")
                return True

    return False

@app.post("/move/")
async def get_best_move(request: MoveRequest):
    fen = request.fen
    board = chess.Board(fen)

    opening_move = find_opening_move(board)
    if opening_move is not None:
        return {"move": opening_move}

    model = NNModel(input_channels, num_res_blocks, num_actions)
    mcts = MCTS(board, args, model)
    model_prediction = mcts.search()
    mcts_probs = model_prediction[0]

    sorted_mcts_probs = sorted(mcts_probs, key=lambda x: x[1], reverse=True)

    print("All moves from MCTS search:")
    for move_prob in sorted_mcts_probs:
        move = chess.Move.from_uci(move_prob[0])
        print(f"Move: {move}, Probability: {move_prob[1]}")

    for move_prob in sorted_mcts_probs:
        move = chess.Move.from_uci(move_prob[0])

        if board.is_capture(move):
            temp_board = board.copy()
            if count_attackers(temp_board, move.to_square, chess.BLACK) > count_defenders(temp_board, move.to_square) == 0:
                print("Black can capture a piece for free with move:", move)
                return {"move": move.uci()}

    # Find the first non-blunder move
    for move_prob in sorted_mcts_probs:
        move = chess.Move.from_uci(move_prob[0])
        if not is_blunder(board, move):
            print("This move is not a blunder: ", move)
            return {"move": move.uci()}

    move = max(sorted_mcts_probs, key=lambda x: x[1])[0]
    return {"move": move}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
