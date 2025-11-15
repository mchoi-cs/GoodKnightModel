#!/usr/bin/env python3
"""
Export trained model weights in various formats.
"""

import torch
import json
import numpy as np
from chess_cnn import create_model


def export_pytorch_weights(model_path='best_chess_model.pth', output_path='model_weights.pth'):
    """
    Export PyTorch weights (standard format).

    Args:
        model_path: Path to trained model
        output_path: Path to save weights
    """
    state_dict = torch.load(model_path, map_location='cpu')
    torch.save(state_dict, output_path)
    print(f"Exported PyTorch weights to {output_path}")


def export_numpy_weights(model_path='best_chess_model.pth', output_dir='weights_numpy'):
    """
    Export weights as numpy arrays (.npz format).
    Useful for loading in other frameworks or languages.

    Args:
        model_path: Path to trained model
        output_dir: Directory to save numpy weights
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    state_dict = torch.load(model_path, map_location='cpu')

    # Convert each tensor to numpy and save
    weights_dict = {}
    for key, tensor in state_dict.items():
        weights_dict[key] = tensor.cpu().numpy()

    # Save all weights in one file
    output_path = f"{output_dir}/model_weights.npz"
    np.savez_compressed(output_path, **weights_dict)
    print(f"Exported numpy weights to {output_path}")

    # Also save individual layers for easier inspection
    for key, array in weights_dict.items():
        layer_path = f"{output_dir}/{key.replace('.', '_')}.npy"
        np.save(layer_path, array)

    print(f"Exported {len(weights_dict)} individual numpy arrays to {output_dir}/")


def export_json_metadata(model_path='best_chess_model.pth', output_path='model_metadata.json'):
    """
    Export model metadata and architecture info as JSON.

    Args:
        model_path: Path to trained model
        output_path: Path to save metadata
    """
    state_dict = torch.load(model_path, map_location='cpu')

    metadata = {
        'num_parameters': sum(p.numel() for p in state_dict.values()),
        'layers': {}
    }

    for key, tensor in state_dict.items():
        metadata['layers'][key] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'num_elements': tensor.numel()
        }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported model metadata to {output_path}")


def export_onnx(model_path='best_chess_model.pth', output_path='model.onnx',
                num_filters=128, num_res_blocks=5):
    """
    Export model to ONNX format (cross-platform).

    Args:
        model_path: Path to trained model
        output_path: Path to save ONNX model
        num_filters: Number of filters (must match trained model)
        num_res_blocks: Number of residual blocks (must match trained model)
    """
    # Create model with same architecture
    model = create_model(num_filters=num_filters, num_res_blocks=num_res_blocks, device='cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 18, 8, 8)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX model to {output_path}")


def export_torchscript(model_path='best_chess_model.pth', output_path='model_scripted.pt',
                       num_filters=128, num_res_blocks=5):
    """
    Export model as TorchScript (portable PyTorch format).

    Args:
        model_path: Path to trained model
        output_path: Path to save TorchScript model
        num_filters: Number of filters (must match trained model)
        num_res_blocks: Number of residual blocks (must match trained model)
    """
    # Create model with same architecture
    model = create_model(num_filters=num_filters, num_res_blocks=num_res_blocks, device='cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create scripted model
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    print(f"Exported TorchScript model to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Export trained model weights')
    parser.add_argument(
        '--model',
        default='best_chess_model.pth',
        help='Path to trained model (default: best_chess_model.pth)'
    )
    parser.add_argument(
        '--format',
        default='all',
        choices=['pytorch', 'numpy', 'onnx', 'torchscript', 'metadata', 'all'],
        help='Export format (default: all)'
    )
    parser.add_argument(
        '--num-filters',
        type=int,
        default=128,
        help='Number of filters (must match trained model)'
    )
    parser.add_argument(
        '--num-res-blocks',
        type=int,
        default=5,
        help='Number of residual blocks (must match trained model)'
    )

    args = parser.parse_args()

    print(f"Exporting model from {args.model}...\n")

    if args.format in ['pytorch', 'all']:
        export_pytorch_weights(args.model, 'model_weights.pth')
        print()

    if args.format in ['numpy', 'all']:
        export_numpy_weights(args.model, 'weights_numpy')
        print()

    if args.format in ['metadata', 'all']:
        export_json_metadata(args.model, 'model_metadata.json')
        print()

    if args.format in ['onnx', 'all']:
        export_onnx(args.model, 'model.onnx', args.num_filters, args.num_res_blocks)
        print()

    if args.format in ['torchscript', 'all']:
        export_torchscript(args.model, 'model_scripted.pt', args.num_filters, args.num_res_blocks)
        print()

    print("Export complete!")


if __name__ == "__main__":
    main()
