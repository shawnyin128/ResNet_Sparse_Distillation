from torch.utils.data import DataLoader, Dataset


def get_dataloader(train_dataset: Dataset,
                   val_dataset: Dataset,
                   batch_size: int=256):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader
