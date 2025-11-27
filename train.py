#!/usr/bin/env python3
"""
Train the chess evaluation CNN on chess positions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob
import os
from chess_cnn import create_model
from pathlib import Path


class ChessDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_files: List of .npz files with data and labels
        """
        print(f"Found {sum(1 for f in data_dir.iterdir())} .npz files")

        # First pass: count total samples
        total_samples = 0
        for file in data_dir.iterdir():
            npz = np.load(file)
            total_samples += len(npz['input'])
            npz.close()

        print(f"Total samples: {total_samples:,}")

        # Pre-allocate arrays
        data_np = np.empty((total_samples, 18, 8, 8), dtype=np.uint8)
        labels_np = np.empty((total_samples,), dtype=np.float32)

        # Second pass: fill arrays
        offset = 0
        for i, file in enumerate(data_dir.iterdir()):
            npz = np.load(file)
            n_samples = len(npz['input'])
            data_np[offset:offset+n_samples] = npz['input']
            labels_np[offset:offset+n_samples] = npz['output']
            offset += n_samples

            if (i + 1) % 50 == 0:
                print(f"  Loaded {i+1}/{sum(1 for f in data_dir.iterdir())} files...")

        print(f"Finished loading all {sum(1 for f in data_dir.iterdir())} files")

        # Convert to PyTorch tensors
        self.data = torch.from_numpy(data_np).float()
        self.labels = torch.tensor(labels_np, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device='cuda'):
    """
    Train the chess evaluation model.

    Args:
        model: ChessEvaluationCNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (tensors, evaluations) in enumerate(train_loader):
            tensors = tensors.to(device)
            evaluations = evaluations.to(device).unsqueeze(1)

            # Check for NaN/Inf in input data
            if torch.isnan(tensors).any() or torch.isinf(tensors).any():
                print(f"Warning: NaN/Inf detected in input tensors at batch {batch_idx}")
                continue
            if torch.isnan(evaluations).any() or torch.isinf(evaluations).any():
                print(f"Warning: NaN/Inf detected in evaluations at batch {batch_idx}")
                continue

            # Forward pass
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, evaluations)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}, skipping batch")
                continue

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for tensors, evaluations in val_loader:
                tensors = tensors.to(device)
                evaluations = evaluations.to(device).unsqueeze(1)

                outputs = model(tensors)
                loss = criterion(outputs, evaluations)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print("-" * 50)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = '/app/models/best_chess_model.pth' if os.path.exists('/app/models') else 'models/best_chess_model.pth'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with val loss: {avg_val_loss:.4f}\n")

    print("Training complete!")


def main():
    import sys

    # Check for directory argument
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_directory>")
        print("Example: python train.py ./training_data")
        return

    data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' does not exist")
        return

    if not data_dir.is_dir():
        print(f"Error: '{data_dir}' is not a directory")
        return

    # Configuration
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.0001  # Lower learning rate to prevent NaN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Creating our datasets
    dataset = ChessDataset(data_dir)
    print(f"Loaded dataset with {len(dataset)} positions")
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model (lightweight for depth-10 evaluations)
    print("\nCreating lightweight model...")
    model = create_model(num_filters=32, num_res_blocks=2, device=device)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train model
    print("\nStarting training...\n")
    train_model(model, train_loader, val_loader=val_loader, num_epochs=num_epochs, lr=learning_rate, device=device)

    model_path = '/app/models/best_chess_model.pth' if os.path.exists('/app/models') else 'models/best_chess_model.pth'
    print(f"\nTraining finished! Best model saved as '{model_path}'")


if __name__ == "__main__":
    main()
