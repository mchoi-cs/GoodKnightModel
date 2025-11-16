#!/usr/bin/env python3
"""
Load chess positions from Hugging Face dataset and convert to tensor format.
Dataset: ssingh22/chess-evaluations
"""

from datasets import load_dataset
from getTensorFromFen import get_tensor_bytes_from_fen
import numpy as np


def main():
    # Load multiple datasets in streaming mode
    print("Loading chess-evaluations datasets in streaming mode...")

    # Combine ALL positions from evals_large, randoms, and tactics
    # evals_large: 13M general positions
    # randoms: 1M random positions
    # tactics: 2.6M tactical positions
    # Total: ~16.6M positions
    print("Loading evals_large (ALL 13M positions)...")
    dataset1 = load_dataset("ssingh22/chess-evaluations", "evals_large", split="train", streaming=True)

    print("Loading randoms (ALL 1M positions)...")
    dataset2 = load_dataset("ssingh22/chess-evaluations", "randoms", split="train", streaming=True)

    print("Loading tactics (ALL 2.6M positions)...")
    dataset3 = load_dataset("ssingh22/chess-evaluations", "tactics", split="train", streaming=True)

    # Interleave datasets for better mixing
    from itertools import chain

    # Use ALL positions from each dataset
    dataset = chain(dataset1, dataset2, dataset3)

    print("Total target: ~16.6M positions from all three subsets")

    batch_size = 10000
    batch_num = 0

    tensors_batch = []
    evaluations_batch = []

    for i, example in enumerate(dataset):
        fen = example['FEN']
        evaluation = example['Evaluation']

        # Convert FEN to tensor bytes
        tensor_bytes = get_tensor_bytes_from_fen(fen)

        # Convert bytes to numpy array (18x8x8)
        tensor = np.frombuffer(tensor_bytes, dtype=np.uint8).reshape(18, 8, 8)

        # Convert evaluation to float (handle mate scores and string formats)
        try:
            if isinstance(evaluation, str):
                # Handle mate scores (M5, M-3, etc.)
                if evaluation.startswith('M') or evaluation.startswith('#'):
                    # Convert mate to large centipawn value
                    mate_num = evaluation.lstrip('M#')
                    if mate_num.startswith('-'):
                        eval_float = -10000.0  # Mate for black (in centipawns)
                    else:
                        eval_float = 10000.0  # Mate for white (in centipawns)
                else:
                    # Regular numeric evaluation (in centipawns)
                    eval_float = float(evaluation.lstrip('+'))
            else:
                eval_float = float(evaluation)

            # Normalize using sigmoid scaling for centipawn values
            # Maps centipawns to (-1, 1) range
            # Formula: 2 / (1 + exp(-x/400)) - 1
            # This gives: 0cp->0, ±200cp->±0.39, ±400cp->±0.62, ±800cp->±0.86, ±10000->±1
            eval_float = 2.0 / (1.0 + np.exp(-eval_float / 400.0)) - 1.0

        except (ValueError, AttributeError):
            # Skip positions with invalid evaluations
            continue

        tensors_batch.append(tensor)
        evaluations_batch.append(eval_float)

        # Save batch when full
        if len(tensors_batch) >= batch_size:
            batch_num += 1
            output_file = f"chess_tensors_batch_{batch_num}.npz"

            np.savez_compressed(
                output_file,
                tensors=np.array(tensors_batch),
                evaluations=np.array(evaluations_batch)
            )

            print(f"Saved batch {batch_num} to {output_file} ({len(tensors_batch)} positions)")

            # Clear batch
            tensors_batch = []
            evaluations_batch = []

    # Save remaining positions
    if tensors_batch:
        batch_num += 1
        output_file = f"chess_tensors_batch_{batch_num}.npz"

        np.savez_compressed(
            output_file,
            tensors=np.array(tensors_batch),
            evaluations=np.array(evaluations_batch)
        )

        print(f"Saved final batch {batch_num} to {output_file} ({len(tensors_batch)} positions)")

    print(f"\nProcessing complete! Created {batch_num} batch files.")


if __name__ == "__main__":
    main()
