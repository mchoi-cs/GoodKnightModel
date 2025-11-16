#!/usr/bin/env python3
"""
CNN model for chess position evaluation.
Input: 18x8x8 tensor (chess position)
Output: evaluation score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessEvaluationCNN(nn.Module):
    """
    Convolutional Neural Network for chess position evaluation.

    Architecture:
    - Input: 18 channels x 8x8 board
    - Multiple convolutional layers to extract spatial features
    - Residual connections for better gradient flow
    - Fully connected layers for final evaluation
    """

    def __init__(self, num_filters=256, num_res_blocks=10, dropout_rate=0.3):
        super(ChessEvaluationCNN, self).__init__()

        # Initial convolution
        self.conv_input = nn.Conv2d(18, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, dropout_rate=dropout_rate) for _ in range(num_res_blocks)
        ])

        # Policy head (for move prediction - optional, can be removed)
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)

        # Value head (for position evaluation)
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_dropout = nn.Dropout(dropout_rate)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 18, 8, 8)

        Returns:
            evaluation: Position evaluation score
        """
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Value head
        val = F.relu(self.value_bn(self.value_conv(x)))
        val = val.view(val.size(0), -1)  # Flatten
        val = F.relu(self.value_fc1(val))
        val = self.value_dropout(val)  # Apply dropout
        val = self.value_fc2(val)  # Output: evaluation score

        return val


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and optional dropout."""

    def __init__(self, num_filters, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


def create_model(num_filters=256, num_res_blocks=10, device='cuda'):
    """
    Create and initialize the chess evaluation model.

    Args:
        num_filters: Number of filters in convolutional layers
        num_res_blocks: Number of residual blocks
        device: 'cuda' or 'cpu'

    Returns:
        model: Initialized ChessEvaluationCNN
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    model = ChessEvaluationCNN(num_filters=num_filters, num_res_blocks=num_res_blocks)
    model = model.to(device)

    return model


def predict_evaluation(model, tensor, device='cuda'):
    """
    Predict evaluation for a single chess position.

    Args:
        model: Trained ChessEvaluationCNN model
        tensor: Chess position tensor (18, 8, 8) as numpy array
        device: 'cuda' or 'cpu'

    Returns:
        evaluation: Predicted evaluation score
    """
    model.eval()

    # Convert to torch tensor and add batch dimension
    x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)

    with torch.no_grad():
        evaluation = model(x)

    return evaluation.item()


if __name__ == "__main__":
    # Example usage
    print("Creating Chess Evaluation CNN...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = create_model(num_filters=128, num_res_blocks=5, device=device)

    # Print model summary
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test with random input
    batch_size = 4
    test_input = torch.randn(batch_size, 18, 8, 8).to(device)

    model.eval()
    with torch.no_grad():
        output = model(test_input)

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze().tolist()}")
