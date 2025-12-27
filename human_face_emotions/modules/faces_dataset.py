"""Dataset module for Human Face Emotions classification.

This module provides PyTorch Lightning DataModule for loading and preprocessing
facial emotion images organized in folder structure by emotion categories.
"""

from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class FacesEmotionDataset(Dataset):
    """PyTorch Dataset for facial emotion images.

    Loads images from a directory structure where each subdirectory
    represents an emotion class.

    Attributes:
        data_dir: Path to the root data directory.
        transform: Torchvision transforms to apply to images.
        images: List of tuples containing (image_path, label_id).
        labels: Sorted list of emotion class names.
        label2id: Mapping from label name to numeric id.
        id2label: Mapping from numeric id to label name.
    """

    def __init__(
        self,
        data_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize the FacesEmotionDataset.

        Args:
            data_dir: Path to the root directory containing emotion subdirectories.
            transform: Optional torchvision transforms to apply to images.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.labels = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}

        self.images = self._load_image_paths()

    def _load_image_paths(self) -> list[Tuple[Path, int]]:
        """Load all image paths and their corresponding labels.

        Returns:
            List of tuples containing (image_path, label_id).
        """
        images = []
        for label in self.labels:
            label_dir = self.data_dir / label
            label_id = self.label2id[label]
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_path in sorted(label_dir.glob(ext)):
                    images.append((img_path, label_id))
        return images

    def __len__(self) -> int:
        """Return the total number of images in the dataset.

        Returns:
            Number of images.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single image and its label.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            Tuple of (image_tensor, label_id).
        """
        img_path, label_id = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_id


class FacesEmotionDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for facial emotion classification.

    Handles data loading, splitting, and preprocessing for training,
    validation, and testing.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the FacesEmotionDataModule.

        Args:
            cfg: Hydra configuration object containing data and training settings.
        """
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data.data_dir)
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.train_val_split = cfg.data.train_val_split
        self.image_size = cfg.data.image_size
        self.seed = cfg.seed
        self.cache_data = cfg.data.cache_data

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.data.normalize.mean,
                    std=cfg.data.normalize.std,
                ),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self._id2label = None
        self._label2id = None
        self._num_classes = None

    @property
    def id2label(self) -> dict[int, str]:
        """Get mapping from id to label name.

        Returns:
            Dictionary mapping numeric ids to label names.
        """
        return self._id2label

    @property
    def label2id(self) -> dict[str, int]:
        """Get mapping from label name to id.

        Returns:
            Dictionary mapping label names to numeric ids.
        """
        return self._label2id

    @property
    def num_classes(self) -> int:
        """Get the number of emotion classes.

        Returns:
            Number of classes.
        """
        return self._num_classes

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        full_dataset = FacesEmotionDataset(
            data_dir=self.data_dir,
            transform=self.transform,
        )

        self._id2label = full_dataset.id2label
        self._label2id = full_dataset.label2id
        self._num_classes = len(full_dataset.labels)

        if self.cache_data:
            full_dataset = self._cache_dataset(full_dataset)

        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    def _cache_dataset(
        self, dataset: FacesEmotionDataset
    ) -> torch.utils.data.TensorDataset:
        """Cache all images and labels in memory as tensors.

        Args:
            dataset: The dataset to cache.

        Returns:
            TensorDataset with all images and labels cached.
        """
        all_images = []
        all_labels = []

        for img, label in tqdm(dataset, desc="Caching data"):
            all_images.append(img)
            all_labels.append(label)

        X = torch.stack(all_images)
        y = torch.tensor(all_labels)

        return torch.utils.data.TensorDataset(X, y)

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.

        Returns:
            DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.

        Returns:
            DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test data loader.

        Returns:
            DataLoader for test data.
        """
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction data loader.

        Returns:
            DataLoader for prediction.
        """
        return self.val_dataloader()
