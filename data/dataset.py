import torchvision

from torch.utils.data import Dataset
from torchvision.transforms import transforms


mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408)
}


std = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761)
}


def get_CIFAR_transform(dataset: str,
                        split: str) -> transforms:
    if dataset == "cifar10" or dataset == "cifar100":
        if split == "train":
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean[dataset], std[dataset])])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean[dataset], std[dataset])])
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def get_dataset(dataset: str) -> tuple[Dataset, Dataset]:
    if dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=f'./{dataset}/data/train',
                                                     train=True,
                                                     download=True,
                                                     transform=get_CIFAR_transform(dataset, "train"))
        val_dataset = torchvision.datasets.CIFAR10(root=f'./{dataset}/data/val',
                                                   train=False,
                                                   download=True,
                                                   transform=get_CIFAR_transform(dataset, "val"))
        return train_dataset, val_dataset
    elif dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root=f'./{dataset}/data/train',
                                                      train=True,
                                                      download=True,
                                                      transform=get_CIFAR_transform(dataset, "train"))
        val_dataset = torchvision.datasets.CIFAR100(root=f'./{dataset}/data/val',
                                                    train=False,
                                                    download=True,
                                                    transform=get_CIFAR_transform(dataset, "val"))
        return train_dataset, val_dataset
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

