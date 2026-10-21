import os

import torch
from torchvision.datasets import SVHN as PyTorchSVHN

DEFAULT_DATA_ROOT = os.path.expanduser("~/data")


class SVHN:
    def __init__(self, preprocess, location=DEFAULT_DATA_ROOT, batch_size=128, num_workers=0):

        modified_location = os.path.join(location, "svhn")

        self.train_dataset = PyTorchSVHN(
            root=modified_location, download=True, split="train", transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchSVHN(
            root=modified_location, download=True, split="test", transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
