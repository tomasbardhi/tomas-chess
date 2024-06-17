import chess
import chess.polyglot

class ChessEngine:
    def __init__(self, polyglot_book_path, board):
        self.polyglot_book_path = polyglot_book_path
        self.board = board
        self.load_opening_book()

    def load_opening_book(self):
        self.opening_book = chess.polyglot.open_reader(self.polyglot_book_path)

    def get_opening_move(self):
        try:
            with self.opening_book:
                entry = self.opening_book.find(self.board)
                return entry.move.uci()
        except IndexError:
            return None

def find_opening_move(board):
    engine = ChessEngine("Titans.bin", board)
    return engine.get_opening_move()
