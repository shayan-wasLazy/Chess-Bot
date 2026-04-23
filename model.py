import numpy as np
import chess
import tensorflow as tf

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model("models/TF_50EPOCHS2.keras")

# =========================
# Move Dictionary (IMPORTANT)
# =========================
# You MUST load the same dictionary used during training

import pickle

with open("move_to_int.pkl", "rb") as f:
    move_to_int = pickle.load(f)

int_to_move = {v: k for k, v in move_to_int.items()}


# =========================
# Board Encoding
# =========================
def board_to_matrix(board):
    matrix = np.zeros((8, 8, 12))

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6

        matrix[row, col, piece_type + piece_color] = 1

    return matrix


# =========================
# Evaluation Function
# =========================
def evaluate(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.2,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    score = 0

    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    # 🔥 Mobility (very important)
    score += 0.1 * len(list(board.legal_moves))

    return score


# =========================
# Minimax
# =========================
def minimax(board, depth, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    legal_moves = list(board.legal_moves)
    best_move = None

    if maximizing:
        max_eval = -float('inf')

        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, False)
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

        return max_eval, best_move

    else:
        min_eval = float('inf')

        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, True)
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

        return min_eval, best_move


# =========================
# Hybrid Prediction (NN + Minimax)
# =========================
def predict_next_move(board):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix, verbose=0)[0]

    legal_moves = list(board.legal_moves)

    move_scores = []

    for move in legal_moves:
        uci = move.uci()
        if uci in move_to_int:
            idx = move_to_int[uci]
            move_scores.append((move, predictions[idx]))
        else:
            move_scores.append((move, -1))  # unseen moves

    # Sort by NN confidence
    move_scores.sort(key=lambda x: x[1], reverse=True)

    best_eval = -float('inf')
    best_move = None

    # Only check top-k moves (speed optimization)
    for move, _ in move_scores[:10]:
        board.push(move)
        eval_score, _ = minimax(board, depth=3, maximizing=False)
        board.pop()

        if eval_score > best_eval:
            best_eval = eval_score
            best_move = move

    return best_move