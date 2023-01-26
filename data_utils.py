
import os
from pathlib import Path
from filelock import FileLock

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_transforms import PILToNumpy, OneHot
from config import DATA_DIR, VALIDATION_FRACTION


def get_trainval_dataset(data_dir):
    tv_ds = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split='trainval',
        target_types='segmentation',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((128, 128)),
            PILToNumpy(),
            OneHot(3, 1)
        ]),
        download=True
    )

    flipped_tv_ds = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split='trainval',
        target_types='segmentation',
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(1.0),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(1.0),
            transforms.Resize((128, 128)),
            PILToNumpy(),
            OneHot(3, 1)
        ]),
        download=True
    )

    trainval_dataset = torch.utils.data.ConcatDataset([tv_ds, flipped_tv_ds])
    return trainval_dataset


def get_test_dataset(data_dir):
    return datasets.OxfordIIITPet(
        root=data_dir,
        split='test',
        target_types='segmentation',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((128, 128)),
            PILToNumpy(),
            OneHot(3, 1)
        ]),
        download=True
    )


def get_datasets(data_dir):
    trainval_dataset = get_trainval_dataset(data_dir)
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainval_dataset,
        [1 - VALIDATION_FRACTION, VALIDATION_FRACTION]
    )
    test_dataset = get_test_dataset(data_dir)
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size):
    with FileLock(os.path.expanduser("~/.data.lock")):
        train_dataset, val_dataset, test_dataset = get_datasets(DATA_DIR)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    return train_dataloader, val_dataloader, test_dataloader


def load_model_state(model, model_filename, device):
    if device == 'cuda':
        model.load_state_dict(torch.load(model_filename))
    elif device == 'cpu':
        model.load_state_dict(torch.load(
            model_filename,
            map_location=torch.device('cpu')
        ))


def download_data(data_dir):
    _ = datasets.OxfordIIITPet(
        root=data_dir,
        split='trainval',
        target_types='segmentation',
        download=True
    )
    _ = datasets.OxfordIIITPet(
        root=data_dir,
        split='test',
        target_types='segmentation',
        download=True
    )

    data_dirpath = Path(data_dir)
    images_dirpath = data_dirpath / 'oxford-iiit-pet/images/'
    masks_dirpath = data_dirpath / 'oxford-iiit-pet/annotations/trimaps/'
    return (
        data_dirpath.exists()
        and images_dirpath.exists()
        and masks_dirpath.exists()
    )


if __name__ == '__main__':
    download_data(DATA_DIR)
