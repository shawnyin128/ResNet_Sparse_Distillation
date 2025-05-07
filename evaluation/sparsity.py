import torch.nn as nn


def calculate_sparsity(model: nn.Module,
                       threshold: float=1e-3) -> float:
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param <= threshold).sum().item()

    sparsity = 100. * zero_params / total_params
    return sparsity