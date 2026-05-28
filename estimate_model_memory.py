"""Estimate model memory usage for spiral models.

This script is intended to provide a reproducible baseline for parameter,
activation, and optimizer memory estimation. It is written as a simple CLI
utility so you can plug in model choices, input dimensions, and precision
assumptions.

Usage example:
    python estimate_model_memory.py --model ResidualCNN_3layers --batch 16 --height 256 --width 256 --f 3 3 --n 64 32
"""

import argparse
import math
import torch
from spiral_deblur_ResNet import (
    ResidualCNN_3layers,
    ResidualCNN_5layers,
    ResidualCNN_5layers_withB0,
    ResidualCNN_5layers_withPhi,
    ResidualCNN_MSResNet_withB0,
)
from spiral_deblur_UNet import MS_UNet, MS_UNet_4stage


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def bytes_for_dtype(num_elements: int, dtype: str) -> int:
    if dtype == 'float16':
        return num_elements * 2
    if dtype == 'float32':
        return num_elements * 4
    if dtype == 'float64':
        return num_elements * 8
    raise ValueError(f'Unsupported dtype: {dtype}')


def estimate_activation_elements(batch: int, channels: int, height: int, width: int) -> int:
    """Estimate the number of elements for a single activation tensor."""
    return batch * channels * height * width


def estimate_simple_activation_memory(batch: int, input_channels: int, height: int, width: int, n: list) -> int:
    """Very rough activation memory estimate for a general conv-based model.

    This function is not exact for every model; it is a fallback for cases
    where a dry forward measurement fails or is unavailable.
    """
    activation_elements = 0

    activation_elements += estimate_activation_elements(batch, input_channels, height, width)
    for ni in n:
        activation_elements += estimate_activation_elements(batch, ni, height, width)
    activation_elements += estimate_activation_elements(batch, 2, height, width)

    # Rough extra storage for backward (saved activations / gradients)
    activation_elements *= 2

    return activation_elements


def measure_activation_elements(model: torch.nn.Module, batch: int, input_channels: int, height: int, width: int) -> int:
    """Run a single forward with hooks to estimate total activation elements.

    This measures the number of elements produced by module outputs during a
    forward pass on CPU. It's more accurate across different architectures
    (UNet vs ResNet) than a hand-rolled formula.
    """
    model_cpu = model.cpu()
    activations = []
    handles = []

    def hook(module, inp, out):
        try:
            # account for tensors and tuples/lists of tensors
            if isinstance(out, torch.Tensor):
                activations.append(out.numel())
            elif isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        activations.append(o.numel())
        except Exception:
            pass

    # register hooks on most modules that produce feature maps
    for m in model_cpu.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Upsample, torch.nn.MaxPool2d, torch.nn.ReLU, torch.nn.Identity)):
            try:
                handles.append(m.register_forward_hook(hook))
            except Exception:
                pass

    with torch.no_grad():
        dummy = torch.zeros((batch, input_channels, height, width))
        try:
            _ = model_cpu(dummy)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

    return sum(activations)


def estimate_optimizer_memory(param_count: int, optimizer: str = 'adam', dtype: str = 'float16') -> int:
    """Estimate optimizer state memory.

    For Adam, we assume 2 additional tensors per parameter.
    For SGD, we assume 1 additional tensor per parameter.
    The dtype is matched to the model parameter dtype so no mixed-precision
    accounting is used in this estimator.
    """
    if optimizer.lower() == 'adam':
        multiplier = 2
    elif optimizer.lower() == 'sgd':
        multiplier = 1
    else:
        multiplier = 1

    return bytes_for_dtype(param_count * multiplier, dtype)


def make_model(model_name: str, f: list, n: list):
    name = model_name
    if name == 'ResidualCNN_3layers':
        return ResidualCNN_3layers(f=f, n=n)
    if name == 'ResidualCNN_5layers':
        return ResidualCNN_5layers(f=f, n=n)
    if name == 'ResidualCNN_5layers_withB0':
        return ResidualCNN_5layers_withB0(f=f, n=n)
    if name == 'ResidualCNN_5layers_withPhi':
        return ResidualCNN_5layers_withPhi(f=f, n=n)
    if name == 'ResidualCNN_MSResNet_withB0':
        return ResidualCNN_MSResNet_withB0(f=f, n=n)
    if name == 'MS_UNet':
        return MS_UNet(f=f, n=n)
    if name == 'MS_UNet_4stage':
        return MS_UNet_4stage(f=f, n=n)
    raise ValueError(f'Unsupported model: {model_name}')


def required_input_channels(model_name: str) -> int:
    if model_name in ('ResidualCNN_3layers', 'ResidualCNN_5layers'):
        return 2
    if model_name == 'ResidualCNN_5layers_withB0':
        return 3
    if model_name == 'ResidualCNN_5layers_withPhi':
        return 4
    if model_name == 'ResidualCNN_MSResNet_withB0':
        return 3
    if model_name in ('MS_UNet', 'MS_UNet_4stage'):
        return 3
    raise ValueError(f'Unsupported model: {model_name}')


def parse_args():
    parser = argparse.ArgumentParser(description='Estimate model memory usage for spiral models.')
    parser.add_argument('--model', type=str, default='ResidualCNN_3layers', help='Model class name',
                        choices=['ResidualCNN_3layers','ResidualCNN_5layers','ResidualCNN_5layers_withB0','ResidualCNN_5layers_withPhi','ResidualCNN_MSResNet_withB0','MS_UNet','MS_UNet_4stage'])
    parser.add_argument('--batch', type=int, default=8, choices=[8, 16, 24, 32], help='Batch size')
    parser.add_argument('--height', type=int, default=256, help='Input height')
    parser.add_argument('--width', type=int, default=256, help='Input width')
    parser.add_argument('--input-channels', type=int, default=3, help='Input channel count')
    parser.add_argument('--output-channels', type=int, default=2, help='Output channel count')
    parser.add_argument('--f', type=int, nargs='+', default=[3, 3], help='Kernel sizes for the model conv layers')
    parser.add_argument('--n', type=int, nargs='+', default=[64, 32], help='Channel counts for the model conv layers')
    parser.add_argument('--param-dtype', type=str, default='float16', choices=['float16', 'float32', 'float64'], help='Parameter storage dtype')
    parser.add_argument('--activation-dtype', type=str, default='float16', choices=['float16', 'float32', 'float64'], help='Activation storage dtype')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type for state estimate')
    return parser.parse_args()

def main():
    args = parse_args()

    model = make_model(args.model, args.f, args.n)
    param_count = count_parameters(model)
    dtype = args.param_dtype
    param_bytes = bytes_for_dtype(param_count, dtype)
    grad_bytes = bytes_for_dtype(param_count, dtype)
    optimizer_bytes = estimate_optimizer_memory(param_count, optimizer=args.optimizer, dtype=dtype)

    required_input_ch = required_input_channels(args.model)
    input_channels = required_input_ch
    if args.input_channels != required_input_ch:
        print(f'Warning: model {args.model} requires {required_input_ch} input channels; using {required_input_ch} instead of {args.input_channels}.')

    # Try measuring activations with a dry forward pass (accurate), fallback to heuristic
    try:
        activation_elements = measure_activation_elements(model, args.batch, input_channels, args.height, args.width)
    except Exception:
        activation_elements = estimate_simple_activation_memory(args.batch, input_channels, args.height, args.width, args.n)

    activation_bytes = bytes_for_dtype(activation_elements, dtype)

    forward_bytes = param_bytes + activation_bytes
    backprop_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes

    print('Model:', args.model)
    print('Batch size:', args.batch)
    print('Input shape: {}x{}x{} (C x H x W)'.format(input_channels, args.height, args.width))
    print('Kernel sizes f:', args.f)
    print('Channel sizes n:', args.n)
    print('Parameter count:', param_count)
    print('Parameter memory:', f'{param_bytes / 1024**2:.2f} MiB ({args.param_dtype})')
    print('Activation memory:', f'{activation_bytes / 1024**2:.2f} MiB ({args.activation_dtype})')
    print('Optimizer state memory:', f'{optimizer_bytes / 1024**2:.2f} MiB (optimizer={args.optimizer})')
    print('Forward estimate:', f'{forward_bytes / 1024**2:.2f} MiB')
    print('Backprop estimate:', f'{backprop_bytes / 1024**2:.2f} MiB')
    print('\nNote: This is a rough estimate and does not include CUDA workspace, data loader buffers, or other library overhead.')


if __name__ == '__main__':
    main()
