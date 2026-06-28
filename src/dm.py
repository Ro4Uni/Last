import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import torch
import pandas as pd
from glob import glob


class MSNITDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = imread(self.images[idx])
        return torch.from_numpy(img), self.labels[idx]


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def generate_df(self, l0, l1, l2, l3, l4):
        return pd.DataFrame({
            'image': l0 + l1 + l2 + l3 + l4,
            'label': [0] * len(l0) + [1] * len(l1) + [2] * len(l2) + [3] * len(l3) + [4] * len(l4)
        })
    def setup(self, stage = None):
        train_0 = glob(str(self.path / 'train' / '0' / '*.png'))
        train_1 = glob(str(self.path / 'train' / '1' / '*.png'))
        train_2 = glob(str(self.path / 'train' / '2' / '*.png'))
        train_3 = glob(str(self.path / 'train' / '3' / '*.png'))
        train_4 = glob(str(self.path / 'train' / '4' / '*.png'))
        self.train_df = self.generate_df(train_0, train_1, train_2, train_3, train_4)
        test_0 = glob(str(self.path / 'test' / '0' / '*.png'))
        test_1 = glob(str(self.path / 'test' / '1' / '*.png'))
        test_2 = glob(str(self.path / 'test' / '2' / '*.png'))
        test_3 = glob(str(self.path / 'test' / '3' / '*.png'))
        test_4 = glob(str(self.path / 'test' / '4' / '*.png'))
        self.test_df = self.generate_df(test_0, test_1, test_2, test_3, test_4)
        self.train_ds = MSNITDataset(self.train_df.image.values, self.train_df.label.values)
        self.test_ds = MSNITDataset(self.test_df.image.values, self.test_df.label.values)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=shuffle
        )