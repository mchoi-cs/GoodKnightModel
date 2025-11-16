#!/usr/bin/env python3
"""
Train the chess evaluation CNN on chess positions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from chess_cnn import create_model


class ChessDataset(Dataset):
    """Dataset for loading chess positions from .npz batch files."""

    def __init__(self, batch_files):
        """
        Args:
            batch_files: List of .npz file paths
        """
        self.data = []

        print(f"Loading {len(batch_files)} batch files...")
        for i, batch_file in enumerate(batch_files):
            data = np.load(batch_file)
            tensors = data['tensors']
            evaluations = data['evaluations']

            for tensor, evaluation in zip(tensors, evaluations):
                self.data.append((tensor, evaluation))

            print(f"Loaded {batch_file}: {len(tensors)} positions (total: {len(self.data)})")

        print(f"Total dataset size: {len(self.data)} positions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor, evaluation = self.data[idx]

        # Convert to torch tensors
        tensor = torch.from_numpy(tensor).float()
        evaluation = torch.tensor(evaluation, dtype=torch.float32)

        return tensor, evaluation


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
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (tensors, evaluations) in enumerate(train_loader):
            tensors = tensors.to(device)
            evaluations = evaluations.to(device).unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, evaluations)

            # Backward pass
            loss.backward()
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
    # Configuration
    batch_size = 128
    num_epochs = 60
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Load batch files
    batch_files = sorted(glob.glob("chess_tensors_batch_*.npz"))

    if not batch_files:
        print("Error: No batch files found. Run load_dataset.py first.")
        return

    print(f"Found {len(batch_files)} batch files")

    # Split into train/val (80/20)
    split_idx = int(len(batch_files) * 0.8)
    train_files = batch_files[:split_idx]
    val_files = batch_files[split_idx:]

    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = ChessDataset(train_files)

    print("\nCreating validation dataset...")
    val_dataset = ChessDataset(val_files)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    print("\nCreating model...")
    model = create_model(num_filters=256, num_res_blocks=20, device=device)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train model
    print("\nStarting training...\n")
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate, device=device)

    model_path = '/app/models/best_chess_model.pth' if os.path.exists('/app/models') else 'models/best_chess_model.pth'
    print(f"\nTraining finished! Best model saved as '{model_path}'")


if __name__ == "__main__":
    main()
