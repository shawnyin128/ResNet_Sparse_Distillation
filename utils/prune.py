import torch
import torch.nn as nn


def weight_prune(model: nn.Module, threshold: float = 1e-3) -> None:
    with torch.no_grad():
        total_params = 0
        total_pruned = 0

        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                mask = param.abs() < threshold
                pruned_count = mask.sum().item()
                param[mask] = 0.0

                total_params += param.numel()
                total_pruned += pruned_count


def activation_prune():
    pass