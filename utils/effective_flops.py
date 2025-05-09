import torch
import torch.nn as nn


def conv_effective_flops(input_tensor: torch.Tensor,
                         conv_layer: nn.Module,
                         threshold: float) -> int:
    with torch.no_grad():
        input_mask = (input_tensor.abs() >= threshold).float()
        input_nonzero = input_mask.sum().item()

        weight = conv_layer.weight.data
        weight_mask = (weight.abs() >= threshold).float()
        weight_nonzero = weight_mask.sum().item()

        C_in = conv_layer.in_channels
        kH, kW = conv_layer.kernel_size
        C_out = conv_layer.out_channels
        stride_h, stride_w = conv_layer.stride
        padding_h, padding_w = conv_layer.padding

        B, _, H_in, W_in = input_tensor.shape
        H_out = (H_in + 2 * padding_h - kH) // stride_h + 1
        W_out = (W_in + 2 * padding_w - kW) // stride_w + 1

        full_flops = B * C_out * H_out * W_out * C_in * kH * kW * 2

        input_ratio = input_nonzero / input_tensor.numel()
        weight_ratio = weight_nonzero / weight.numel()

        effective_flops = int(full_flops * input_ratio * weight_ratio)
        return effective_flops


def linear_effective_flops(input_tensor: torch.Tensor,
                           linear_layer: nn.Module,
                           threshold: float) -> int:
    with torch.no_grad():
        input_mask = (input_tensor.abs() >= threshold).float()
        input_nonzero = input_mask.sum().item()

        weight = linear_layer.weight.data
        weight_mask = (weight.abs() >= threshold).float()
        weight_nonzero = weight_mask.sum().item()

        full_flops = input_tensor.shape[0] * linear_layer.in_features * linear_layer.out_features * 2
        input_ratio = input_nonzero / input_tensor.numel()
        weight_ratio = weight_nonzero / weight.numel()

        return int(full_flops * input_ratio * weight_ratio)
