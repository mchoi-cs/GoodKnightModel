import sys
import numpy as np
# Get a 18 x 8 x 8 tensor representation of a chess position from a FEN string
# 0-11: 6 piece types x 2 for each color
# 12-15: Castling rights
# 16: Enpassant target square
# 17: Active player (1: White to move, 0: Black to move)

NUM_BYTES = 18 * 8 * 8

def get_tensor_bytes_from_fen(fen: str) -> bytes:
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((18, 8, 8), dtype=np.uint8)

    fen_parts = fen.split(' ')
    board_part = fen_parts[0]
    active_player = fen_parts[1] if len(fen_parts) > 1 else 'w'
    castling_part = fen_parts[2] if len(fen_parts) > 2 else '-'
    enpassant_part = fen_parts[3] if len(fen_parts) > 3 else '-'

    rows = board_part.split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                if char in piece_to_index:
                    tensor[piece_to_index[char], r, c] = 1
                c += 1

    # Decode castling rights (channels 12-15)
    if castling_part != '-':
        if 'K' in castling_part:
            tensor[12, :, :] = 1
        if 'Q' in castling_part:
            tensor[13, :, :] = 1
        if 'k' in castling_part:
            tensor[14, :, :] = 1
        if 'q' in castling_part:
            tensor[15, :, :] = 1

    # If en passant is available, mark the specific square on channel 16
    if enpassant_part != '-':
        col = ord(enpassant_part[0]) - ord('a')
        row = 8 - int(enpassant_part[1])
        tensor[16, row, col] = 1

    # Decode active player (channel 17)
    if active_player == 'w':
        tensor[17, :, :] = 1
    # 1: White to move
    # else: black to move (already 0 by default)

    # Return as a Python bytes object
    return tensor.tobytes()

def main():
    args = sys.argv
    if len(args) <= 1:
        print("Missing FEN argument")
        return
    fen = args[1]

    tensor_bytes = get_tensor_bytes_from_fen(fen)
    hex_str = tensor_bytes.hex()
    # Ensure the hex string is padded to the expected length (NUM_BYTES * 2 hex chars)
    expected_hex_length = NUM_BYTES * 2
    print(hex_str.zfill(expected_hex_length))

if __name__ == "__main__":
    main()

