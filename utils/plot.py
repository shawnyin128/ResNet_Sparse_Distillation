import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_prune_sparsity_metrics(thresholds, accuracy_lists, sparsity_lists, model_names):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    num_models = len(model_names)
    colors = cm.get_cmap('tab10', num_models)

    for idx, (acc_list, name) in enumerate(zip(accuracy_lists, model_names)):
        ax1.plot(thresholds, acc_list, label=f'{name} Acc', color=colors(idx), linestyle='-')
    ax1.set_xlabel("Prune Threshold")
    ax1.set_ylabel("Accuracy")
    ax1.set_xscale("log")
    ax1.tick_params(axis='y')
    ax1.grid(False)

    ax2 = ax1.twinx()
    for idx, (sparsity_list, name) in enumerate(zip(sparsity_lists, model_names)):
        ax2.plot(thresholds, sparsity_list, label=f'{name} Sparsity', color=colors(idx), linestyle='--')
    ax2.set_ylabel("Sparsity")
    ax2.tick_params(axis='y')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.title("Threshold vs Accuracy & Sparsity")
    plt.tight_layout()
    plt.show()

def plot_prune_flops_metrics(thresholds, accuracy_lists, flops_lists, model_names):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    num_models = len(model_names)
    colors = cm.get_cmap('tab10', num_models)

    for idx, (acc_list, name) in enumerate(zip(accuracy_lists, model_names)):
        ax1.plot(thresholds, acc_list, label=f'{name} Acc', color=colors(idx), linestyle='-')
    ax1.set_xlabel("Prune Threshold")
    ax1.set_ylabel("Accuracy")
    ax1.set_xscale("log")
    ax1.tick_params(axis='y')
    ax1.grid(False)

    ax2 = ax1.twinx()
    for idx, (flops_list, name) in enumerate(zip(flops_lists, model_names)):
        ax2.plot(thresholds, flops_list, label=f'{name} FLOPs', color=colors(idx), linestyle='--')
    ax2.set_ylabel("FLOPs")
    ax2.tick_params(axis='y')
    ax2.set_yscale("log")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.title("Threshold vs Accuracy & FLOPs")
    plt.tight_layout()
    plt.show()