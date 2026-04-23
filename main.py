import streamlit as st
import chess
import chess.svg
from streamlit.components.v1 import html
import tensorflow as tf



# Load your model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/TF_50EPOCHS2.keras")

model = load_model()

# import your functions
# make sure predict_next_move and board_to_matrix are accessible
from model import predict_next_move

# Initialize board
if "board" not in st.session_state:
    st.session_state.board = chess.Board()

board = st.session_state.board

st.title("♟️ Chess AI")

# Display board
board_svg = chess.svg.board(board=board, size=500)
st.image(board_svg)

# User input
move_input = st.text_input("Enter your move (e.g. e2e4):", key="move_input")

if st.button("Play Move"):
    try:
        move = chess.Move.from_uci(move_input)

        if move in board.legal_moves:
            st.session_state.board.push(move)

            if not st.session_state.board.is_game_over():
                ai_move = predict_next_move(st.session_state.board)
                st.session_state.board.push(ai_move)
                st.success(f"AI played: {ai_move.uci()}")

            # Clear input
            # st.session_state.move_input = ""

            st.rerun()

        else:
            st.error("Illegal move")

    except ValueError:
        st.error("Invalid move format")

# Game status
if board.is_game_over():
    st.write("Game Over:", board.result())