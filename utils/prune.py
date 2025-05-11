import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.effective_flops import conv_effective_flops, linear_effective_flops


def block_weight_prune_l2(model: nn.Module,
                          prune_ratio: float = 1e-3) -> None:
    with torch.no_grad():
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential) and "downsample" not in name:
                for submodule in module.modules():
                    if isinstance(submodule, nn.Conv2d):
                        weight = submodule.weight.data
                        flat = weight.view(-1).abs()
                        k = int(flat.numel() * prune_ratio)
                        if k > 0:
                            threshold = torch.kthvalue(flat, k).values.item()
                            mask = weight.abs() < threshold
                            weight[mask] = 0.0


def weight_prune(model: nn.Module,
                 threshold: float = 1e-3) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                mask = param.abs() < threshold
                param[mask] = 0.0



def activation_prune(model: nn.Module,
                     x: torch.Tensor,
                     threshold: float) -> tuple:
    total_flops = 0
    batch_size = x.shape[0]

    x_in = x
    x = model.conv1(x)
    total_flops += conv_effective_flops(x_in, model.conv1, threshold)
    x = model.bn1(x)
    x = F.relu(x)

    x = torch.where(x.abs() < threshold, torch.zeros_like(x), x)

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            x_pruned = torch.where(x.abs() < threshold, torch.zeros_like(x), x)

            out = block.conv1(x_pruned)
            total_flops += conv_effective_flops(x_pruned, block.conv1, threshold)
            out = block.bn1(out)
            out = F.relu(out)

            out = torch.where(out.abs() < threshold, torch.zeros_like(out), out)

            out2 = block.conv2(out)
            total_flops += conv_effective_flops(out, block.conv2, threshold)
            out2 = block.bn2(out2)

            if block.shortcut is not None and len(block.shortcut) > 0:
                shortcut = block.shortcut(x_pruned)
                total_flops += conv_effective_flops(x_pruned, block.shortcut[0], threshold)
            else:
                shortcut = x_pruned

            x = F.relu(out2 + shortcut)

            x = torch.where(x.abs() < threshold, torch.zeros_like(x), x)

    x = F.avg_pool2d(x, 4)
    x = x.view(x.size(0), -1)

    x = torch.where(x.abs() < threshold, torch.zeros_like(x), x)
    total_flops += linear_effective_flops(x, model.linear, threshold)
    x = model.linear(x)

    avg_flops = total_flops / batch_size
    return x, avg_flops
