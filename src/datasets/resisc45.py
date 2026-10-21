import abc
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader

DEFAULT_DATA_ROOT = os.path.expanduser("~/data")


# modified from: https://github.com/microsoft/torchgeo
class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Base class for pre-cropped image datasets without geospatial metadata."""

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the image and label at the given index."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the sample count."""

    def __str__(self) -> str:
        """Return the dataset type and sample count."""
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class VisionClassificationDataset(VisionDataset, ImageFolder):
    """Class-directory images with an optional image transform."""

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Load class directories, optionally filtering files before indexing."""
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # ImageFolder initializes this attribute; set the image transform afterward.
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return a transformed image and its class label."""
        image, label = self._load_image(index)

        if self.transforms is not None:
            return self.transforms(image), label

        return image, label

    def __len__(self) -> int:
        """Return the sample count."""
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load an image and convert its class index to a tensor."""
        img, label = ImageFolder.__getitem__(self, index)
        label = torch.tensor(label)
        return img, label


class RESISC45Dataset(VisionClassificationDataset):
    """RESISC45 images filtered by an explicit train/val/test filename list.

    Dataset: https://doi.org/10.1109/jproc.2017.2675998
    Split definitions: https://arxiv.org/abs/1911.06721
    """

    directory = "resisc45/NWPU-RESISC45"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",
    }
    split_md5s = {
        "train": "b5a4c05a37de15e4ca886696a85c403e",
        "val": "a0770cee4c5ca20b8c32bbd61e114805",
        "test": "3dda9e4988b47eb1de9f07993653eb08",
    }
    classes = [
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Read images under ``root/resisc45/NWPU-RESISC45`` for the requested split."""
        assert split in self.splits
        self.root = root

        valid_fns = set()
        with open(os.path.join(self.root, "resisc45", f"resisc45-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            is_valid_file=is_in_split,
        )


class RESISC45:
    def __init__(self, preprocess, location=DEFAULT_DATA_ROOT, batch_size=32, num_workers=0):

        self.train_dataset = RESISC45Dataset(root=location, split="train", transforms=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = RESISC45Dataset(root=location, split="test", transforms=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
        )

        self.classnames = [" ".join(c.split("_")) for c in RESISC45Dataset.classes]
