
import os
from pathlib import Path
from glob import glob
from filelock import FileLock
import pandas as pd

import torch
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_transforms import PILToNumpy, OneHot
from config import DATA_DIR, IMAGE_DIR, MASK_DIR, VALIDATION_FRACTION


class ImageMaskDataset(VisionDataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_list=None, # list of images to include, no file extension
        transform=None,
        target_transform=None,
        loader=default_loader
    ):
        super().__init__(
            image_dir,
            transform=transform,
            target_transform=target_transform
        )

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.loader = loader
        self.image_set = None
        if image_list:
            self.image_set = set(image_list)

        self.images = self.find_images(self.image_dir, self.image_set)
        self.masks = self.find_masks(self.mask_dir, self.images)

    def find_images(self, image_dir, image_set=None):
        image_dir = os.path.expanduser(image_dir)
        image_dirpath = Path(image_dir)

        images = []
        for image_filename in os.listdir(image_dir):
            image_path = image_dirpath / image_filename
            if image_set and image_path.stem not in image_set:
                continue
            if image_path.suffix in IMG_EXTENSIONS:
                images.append(str(image_path))
        return images

    def find_masks(self, mask_dir, images):
        mask_dir = os.path.expanduser(mask_dir)
        mask_dirpath = Path(mask_dir)

        masks = []
        for image_path in images:
            image_filename = Path(image_path).stem
            for ext in IMG_EXTENSIONS:
                mask_path = (mask_dirpath / image_filename).with_suffix(ext)
                if mask_path.exists():
                    masks.append(str(mask_path))
                    break
            else:
                raise FileNotFoundError(
                    'Could not find mask for image {}'.format(image_path)
                )
        return masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = self.loader(image_path)
        mask  = self.loader(mask_path)

        if self.transform is not None:
            sample = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sample, mask



def get_trainval_dataset():
    trainval_list = pd.read_csv(
        open(
            os.path.join(
                DATA_DIR,
                'oxford-iiit-pet/annotations/trainval.txt'
            ),
            'r'
        ),
        header=None,
        index_col=None,
        delimiter=' '
    )[0].tolist()
    tv_ds = ImageMaskDataset(
        IMAGE_DIR,
        MASK_DIR,
        image_list=trainval_list,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((128, 128)),
            PILToNumpy(),
            OneHot(3, 1)
        ])
    )

    flipped_tv_ds = ImageMaskDataset(
        IMAGE_DIR,
        MASK_DIR,
        image_list=trainval_list,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(1.0),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((128, 128)),
            PILToNumpy(),
            OneHot(3, 1)
        ])
    )

    trainval_dataset = torch.utils.data.ConcatDataset([tv_ds, flipped_tv_ds])
    return trainval_dataset


def get_test_dataset():
    test_list = pd.read_csv(
        open(
            os.path.join(
                DATA_DIR,
                'oxford-iiit-pet/annotations/test.txt'
            ),
            'r'
        ),
        header=None,
        index_col=None,
        delimiter=' '
    )[0].tolist()
    return ImageMaskDataset(
        IMAGE_DIR,
        MASK_DIR,
        image_list=test_list,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((128, 128)),
            PILToNumpy(),
            OneHot(3, 1)
        ])
    )


def get_datasets():
    trainval_dataset = get_trainval_dataset()
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainval_dataset,
        [1 - VALIDATION_FRACTION, VALIDATION_FRACTION]
    )
    test_dataset = get_test_dataset()
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size):
    train_dataset, val_dataset, test_dataset = get_datasets()

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
