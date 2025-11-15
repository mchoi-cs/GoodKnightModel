#!/usr/bin/env python3
"""
Load chess positions from Hugging Face dataset and convert to tensor format.
Dataset: ssingh22/chess-evaluations
"""

from datasets import load_dataset
from getTensorFromFen import get_tensor_bytes_from_fen
import numpy as np


def main():
    # Load the tactics dataset in streaming mode
    print("Loading chess-evaluations dataset (tactics) in streaming mode...")
    dataset = load_dataset("ssingh22/chess-evaluations", "tactics", split="train", streaming=True)

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

        tensors_batch.append(tensor)
        evaluations_batch.append(evaluation)

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
