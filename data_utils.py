
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_transforms import PILToNumpy, OneHot
from config import DATA_DIR, BATCH_SIZE, VALIDATION_FRACTION


def get_trainval_dataset():
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


def get_test_dataset():
    return datasets.OxfordIIITPet(
        root=DATA_DIR,
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


def get_datasets():
    trainval_dataset = get_trainval_dataset()
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainval_dataset,
        [1 - VALIDATION_FRACTION, VALIDATION_FRACTION]
    )
    test_dataset = get_test_dataset()
    return train_dataset, val_dataset, test_dataset


def get_dataloaders():
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
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
