import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils.prune import weight_prune, activation_prune
from evaluation.validate import validate
from evaluation.sparsity import calculate_sparsity


def generate_thresholds(levels: list) -> list:
    thresholds = []
    for level in levels:
        for i in range(10):
            for j in range(10):
                thresholds.append(((i+1) + (j+1)/10) * level)
    return thresholds


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

    for i in range(len(thresholds)):
        threshold = thresholds[i]
        total_flops = 0.0
        total_accuracy = 0.0
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
            total_accuracy += (100. * correct / total)

            pbar.set_postfix({"accuracy": (100. * correct / total), "flops": flops})

        accuracy_list.append(total_accuracy / total_iteration)
        flops_list.append(total_flops / total_iteration)

    return accuracy_list, flops_list
