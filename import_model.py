#!/usr/bin/env python3
"""
Import and load model weights from various formats.
"""

import torch
import numpy as np
from chess_cnn import create_model
from getTensorFromFen import get_tensor_bytes_from_fen


def load_pytorch_weights(weights_path='model_weights.pth', num_filters=256, num_res_blocks=10, device='cpu'):
    """
    Load PyTorch weights (.pth format).

    Args:
        weights_path: Path to PyTorch weights file
        num_filters: Number of filters (must match exported model)
        num_res_blocks: Number of residual blocks (must match exported model)
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model ready for inference
    """
    model = create_model(num_filters=num_filters, num_res_blocks=num_res_blocks, device=device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded PyTorch model from {weights_path}")
    return model


def load_numpy_weights(weights_path='weights_numpy/model_weights.npz', num_filters=256, num_res_blocks=10, device='cpu'):
    """
    Load weights from NumPy format (.npz).

    Args:
        weights_path: Path to NumPy weights file
        num_filters: Number of filters (must match exported model)
        num_res_blocks: Number of residual blocks (must match exported model)
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model ready for inference
    """
    # Create model
    model = create_model(num_filters=num_filters, num_res_blocks=num_res_blocks, device=device)

    # Load numpy weights
    weights = np.load(weights_path)

    # Convert numpy arrays to torch tensors and load into model
    state_dict = {}
    for key in weights.files:
        state_dict[key] = torch.from_numpy(weights[key])

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded NumPy model from {weights_path}")
    return model


def load_onnx_model(model_path='model.onnx'):
    """
    Load ONNX model for inference.

    Args:
        model_path: Path to ONNX model file

    Returns:
        onnx_session: ONNX runtime inference session
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Install with: pip install onnxruntime")
        return None

    session = ort.InferenceSession(model_path)
    print(f"Loaded ONNX model from {model_path}")
    return session


def load_torchscript_model(model_path='model_scripted.pt', device='cpu'):
    """
    Load TorchScript model.

    Args:
        model_path: Path to TorchScript model file
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded TorchScript model
    """
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print(f"Loaded TorchScript model from {model_path}")
    return model


def predict_from_fen(model, fen, device='cpu', model_type='pytorch'):
    """
    Predict evaluation from FEN string.

    Args:
        model: Loaded model (PyTorch, TorchScript, or ONNX session)
        fen: FEN string of chess position
        device: 'cuda' or 'cpu'
        model_type: 'pytorch', 'torchscript', or 'onnx'

    Returns:
        evaluation: Predicted evaluation score
    """
    # Convert FEN to tensor
    tensor_bytes = get_tensor_bytes_from_fen(fen)
    tensor = np.frombuffer(tensor_bytes, dtype=np.uint8).reshape(18, 8, 8)

    if model_type == 'onnx':
        # ONNX inference
        input_tensor = tensor.astype(np.float32)[np.newaxis, :]  # Add batch dimension
        inputs = {model.get_inputs()[0].name: input_tensor}
        outputs = model.run(None, inputs)
        evaluation = outputs[0][0][0]
    else:
        # PyTorch or TorchScript inference
        input_tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        evaluation = output.item()

    return evaluation


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Import and test model weights')
    parser.add_argument(
        '--format',
        default='pytorch',
        choices=['pytorch', 'numpy', 'onnx', 'torchscript'],
        help='Import format (default: pytorch)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to weights file (optional, uses defaults)'
    )
    parser.add_argument(
        '--num-filters',
        type=int,
        default=256,
        help='Number of filters (must match trained model)'
    )
    parser.add_argument(
        '--num-res-blocks',
        type=int,
        default=10,
        help='Number of residual blocks (must match trained model)'
    )
    parser.add_argument(
        '--test-fen',
        type=str,
        default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        help='FEN string to test prediction'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )

    args = parser.parse_args()

    # Determine device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}\n")

    # Load model based on format
    if args.format == 'pytorch':
        weights_path = args.weights or 'model_weights.pth'
        model = load_pytorch_weights(weights_path, args.num_filters, args.num_res_blocks, device)
        model_type = 'pytorch'

    elif args.format == 'numpy':
        weights_path = args.weights or 'weights_numpy/model_weights.npz'
        model = load_numpy_weights(weights_path, args.num_filters, args.num_res_blocks, device)
        model_type = 'pytorch'

    elif args.format == 'onnx':
        weights_path = args.weights or 'model.onnx'
        model = load_onnx_model(weights_path)
        model_type = 'onnx'
        if model is None:
            return

    elif args.format == 'torchscript':
        weights_path = args.weights or 'model_scripted.pt'
        model = load_torchscript_model(weights_path, device)
        model_type = 'torchscript'

    # Test prediction
    print(f"\nTesting prediction on FEN: {args.test_fen}")
    evaluation = predict_from_fen(model, args.test_fen, device, model_type)
    print(f"Predicted evaluation: {evaluation:.4f}")

    # Test a few standard positions
    print("\n" + "="*60)
    print("Testing on standard positions:")
    print("="*60)

    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Scholar's mate", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    ]

    for name, fen in test_positions:
        try:
            eval_score = predict_from_fen(model, fen, device, model_type)
            print(f"{name:20s}: {eval_score:8.4f}")
        except Exception as e:
            print(f"{name:20s}: Error - {e}")

    print("\nModel loaded successfully and ready for inference!")


if __name__ == "__main__":
    main()
