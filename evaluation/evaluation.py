from tqdm.notebook import tqdm

from utils.prune import weight_prune
from evaluation.validate import validate
from evaluation.sparsity import calculate_sparsity


def generate_thresholds(levels):
    thresholds = []
    for level in levels:
        for i in range(10):
            for j in range(10):
                thresholds.append(((i+1) + (j+1)/10) * level)
    return thresholds


def evaluate_weights_prune(model, thresholds, val_dataloader):
    accuracy_list = []
    sparsity_list = []
    for threshold in tqdm(thresholds, leave=False):
        weight_prune(model, threshold)
        accuracy_list.append(validate(model, val_dataloader))
        sparsity_list.append(calculate_sparsity(model, threshold))

    return accuracy_list, sparsity_list
