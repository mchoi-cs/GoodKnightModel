# GoodKnight Model Training

This repository contains the training code for the GoodKnight chess evaluation model, a lightweight CNN that learns to evaluate chess positions.

## Related Projects

- **Main Project**: [GoodKnight](https://github.com/choiIsabelle/GoodKnight) - The complete chess application
- **Data Generator**: [GoodKnightDataGenerator](https://github.com/EricHayter/GoodKnightDataGenerator) - Generates training data from chess positions

## Overview

The GoodKnight model is a Convolutional Neural Network (CNN) designed to evaluate chess positions. It takes an 18x8x8 tensor representation of a chess board as input and outputs a numerical evaluation score.

### Model Architecture

- **Input**: 18 channels x 8x8 board representation
  - 12 channels for piece positions (6 piece types x 2 colors)
  - 6 channels for additional features (castling rights, en passant, turn, etc.)
- **Architecture**:
  - Initial convolutional layer with batch normalization
  - Configurable residual blocks (default: 2 blocks with 32 filters)
  - Value head with dropout for regularization
  - Final fully connected layers producing evaluation score
- **Output**: Single evaluation score (positive favors white, negative favors black)

The model uses MSE loss and is trained on chess positions evaluated at depth 10.

## Training Process

### Data Format

Training data should be in `.npz` format with the following structure:
- `input`: Chess position tensors (shape: N x 18 x 8 x 8, dtype: uint8)
- `output`: Evaluation scores (shape: N, dtype: float32)

Use the [GoodKnightDataGenerator](https://github.com/EricHayter/GoodKnightDataGenerator) to generate this training data.

### Training Features

- **80/20 train/validation split**
- **Adam optimizer** with L2 regularization (weight_decay=1e-4)
- **Gradient clipping** to prevent exploding gradients
- **Batch normalization** for stable training
- **Dropout** for regularization (20% dropout rate)
- **Early stopping**: Saves the best model based on validation loss
- **NaN detection**: Automatically skips batches with NaN/Inf values

### Model Configuration

Default lightweight configuration:
- **Filters**: 32 convolutional filters
- **Residual Blocks**: 2
- **Parameters**: ~50,000 trainable parameters
- **Batch Size**: 128
- **Learning Rate**: 0.0001
- **Epochs**: 20

## Running Locally

### Prerequisites

- Python 3.11 or higher
- (Optional) CUDA-capable GPU for faster training

### Setup

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/GoodKnightModel.git
cd GoodKnightModel
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

For CPU-only training:
```bash
pip install -r requirements-cpu.txt
```

For GPU training (requires CUDA):
```bash
pip install -r requirements-gpu.txt
```

### Generating Training Data

Before training, you need to generate training data using the [GoodKnightDataGenerator](https://github.com/EricHayter/GoodKnightDataGenerator):

```bash
# Clone the data generator
git clone https://github.com/EricHayter/GoodKnightDataGenerator.git
cd GoodKnightDataGenerator

# Follow the instructions in that repository to generate .npz files
# The data will be saved in a directory (e.g., ./training_data)
```

### Training the Model

Once you have your training data in a directory (e.g., `./training_data`), run:

```bash
python train.py <path_to_training_data>
```

Example:
```bash
python train.py ./training_data
```

The training script will:
1. Load all `.npz` files from the specified directory
2. Split data into training (80%) and validation (20%) sets
3. Train the model for 20 epochs
4. Save the best model based on validation loss to `models/best_chess_model.pth`

### Monitoring Training

During training, you'll see:
- Batch-level loss every 100 batches
- Epoch-level training and validation loss
- Notifications when a new best model is saved

Example output:
```
Using device: cuda
Found 50 .npz files
Total samples: 1,000,000
Loaded dataset with 1,000,000 positions

Creating lightweight model...
Model has 52,832 parameters

Starting training...

Epoch [1/20], Batch [100/6250], Loss: 2.3456
...
Epoch [1/20]
  Train Loss: 2.1234
  Val Loss:   2.0987
--------------------------------------------------
Saved new best model with val loss: 2.0987
```

### Using the Trained Model

After training, you can load and use the model:

```python
import torch
from chess_cnn import create_model

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(num_filters=32, num_res_blocks=2, device=device)
model.load_state_dict(torch.load('models/best_chess_model.pth'))
model.eval()

# Evaluate a position (assuming you have a position tensor)
# position_tensor shape: (18, 8, 8)
import numpy as np
from chess_cnn import predict_evaluation

evaluation = predict_evaluation(model, position_tensor, device=device)
print(f"Position evaluation: {evaluation}")
```

## Docker Support

You can also run training in a Docker container (note: requires updating Dockerfile for current setup):

```bash
docker build -t goodknight-training .
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models goodknight-training
```

## Project Structure

```
.
├── chess_cnn.py          # CNN model architecture
├── train.py              # Training script
├── requirements-cpu.txt  # CPU-only dependencies
├── requirements-gpu.txt  # GPU dependencies
├── Dockerfile           # Docker configuration
└── README.md            # This file
```

## Hardware Requirements

### Minimum
- CPU: Modern multi-core processor
- RAM: 8GB
- Storage: 10GB for data and models

### Recommended
- GPU: NVIDIA GPU with 6GB+ VRAM (for faster training)
- RAM: 16GB+
- Storage: 50GB+ for larger datasets

## License

See the main [GoodKnight project](https://github.com/choiIsabelle/GoodKnight) for license information.

## Contributing

Please submit issues and pull requests to the main [GoodKnight repository](https://github.com/choiIsabelle/GoodKnight).
