import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils.prune import weight_prune, activation_prune
from evaluation.validate import validate
from evaluation.sparsity import calculate_sparsity


def evaluate_weights_prune(model: nn.Module,
                           thresholds: list,
                           val_dataloader: DataLoader) -> tuple:
    accuracy_list = []
    sparsity_list = []
    for threshold in tqdm(thresholds, leave=False):
        weight_prune(model, threshold)
        accuracy_list.append(validate(model, val_dataloader))
        sparsity_list.append(calculate_sparsity(model, threshold))

    return accuracy_list, sparsity_list


def evaluate_activation_prune(model: nn.Module,
                              thresholds: list,
                              val_dataloader: DataLoader,
                              device: str="cuda") -> tuple:
    model.to(device)
    model.eval()

    accuracy_list = []
    flops_list = []

    for threshold in tqdm(thresholds, leave=False):
        total_flops = 0.0
        total_correct = 0
        total_samples = 0
        total_iteration = 0
        pbar = tqdm(val_dataloader, desc=f"T:{threshold}", leave=False)
        for data in pbar:
            total_iteration += 1
            x, y = data
            x, y = x.to(device), y.to(device)

            output, flops = activation_prune(model, x, threshold)

            total_flops += flops

            correct = 0
            total = 0
            _, predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            total_correct += correct
            total_samples += total

            pbar.set_postfix({"accuracy": (100. * correct / total), "flops": flops})

        accuracy_list.append(100. * total_correct / total_samples)
        flops_list.append(total_flops / total_iteration)

    return accuracy_list, flops_list
