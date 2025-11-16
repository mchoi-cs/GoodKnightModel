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

    # Combine evals_large, randoms, and tactics for comprehensive training
    # evals_large: 13M general positions
    # randoms: 1M random positions
    # tactics: 2.6M tactical positions
    print("Loading evals_large (13M positions)...")
    dataset1 = load_dataset("ssingh22/chess-evaluations", "evals_large", split="train", streaming=True)

    print("Loading randoms (1M positions)...")
    dataset2 = load_dataset("ssingh22/chess-evaluations", "randoms", split="train", streaming=True)

    print("Loading tactics (2.6M positions)...")
    dataset3 = load_dataset("ssingh22/chess-evaluations", "tactics", split="train", streaming=True)

    # Interleave datasets for better mixing
    from itertools import chain, islice

    # Take samples from each dataset
    dataset1_sample = islice(dataset1, 500_000)  # 500k from evals_large
    dataset2_sample = islice(dataset2, 200_000)  # 200k from randoms
    dataset3_sample = islice(dataset3, 300_000)  # 300k from tactics

    # Chain them together (total: 1M positions)
    dataset = chain(dataset1_sample, dataset2_sample, dataset3_sample)

    print("Total target: ~1M positions from multiple subsets")

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
                    # Convert mate to capped evaluation (positive for white, negative for black)
                    # Cap at ±15 to prevent huge loss values
                    mate_num = evaluation.lstrip('M#')
                    if mate_num.startswith('-'):
                        eval_float = -15.0  # Mate for black
                    else:
                        eval_float = 15.0  # Mate for white
                else:
                    # Regular numeric evaluation, remove any + prefix
                    eval_float = float(evaluation.lstrip('+'))
            else:
                eval_float = float(evaluation)

            # Clip evaluations to reasonable range (-15 to +15 pawns)
            eval_float = np.clip(eval_float, -15.0, 15.0)

            # Normalize to [-1, 1] range for better training
            eval_float = eval_float / 15.0

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
