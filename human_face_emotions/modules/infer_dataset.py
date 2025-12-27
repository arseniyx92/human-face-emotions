"""Inference dataset module for Human Face Emotions classification.

This module provides PyTorch Lightning DataModule for loading images
for inference without labels.
"""

from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class InferDataset(Dataset):
    """PyTorch Dataset for inference on facial images.

    Loads images from a directory without requiring label information.

    Attributes:
        image_paths: List of paths to images.
        transform: Torchvision transforms to apply to images.
    """

    def __init__(
        self,
        data_path: Path | str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 48,
    ) -> None:
        """Initialize the InferDataset.

        Args:
            data_path: Path to directory with images or single image file.
            transform: Optional torchvision transforms.
            image_size: Size to resize images to. Defaults to 48.
        """
        self.data_path = Path(data_path)
        self.image_size = image_size

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.image_paths = self._collect_image_paths()

    def _collect_image_paths(self) -> list[Path]:
        """Collect all image paths from the data directory.

        Returns:
            List of paths to image files.

        Raises:
            ValueError: If no images found or path doesn't exist.
        """
        if not self.data_path.exists():
            raise ValueError(f"Path does not exist: {self.data_path}")

        if self.data_path.is_file():
            if self._is_image_file(self.data_path):
                return [self.data_path]
            raise ValueError(f"File is not an image: {self.data_path}")

        image_paths = []
        for file_path in sorted(self.data_path.glob("**/*")):
            if file_path.is_file() and self._is_image_file(file_path):
                image_paths.append(file_path)

        if not image_paths:
            raise ValueError(f"No images found in: {self.data_path}")

        return image_paths

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        """Check if a file is an image based on extension.

        Args:
            path: Path to check.

        Returns:
            True if file has image extension.
        """
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    def __len__(self) -> int:
        """Return the total number of images.

        Returns:
            Number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single image for inference.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            Dictionary with 'pixel_values' tensor and 'image_path' string.
        """
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "image_path": str(img_path),
        }


class InferDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for inference.

    Handles data loading for model inference without labels.
    """

    def __init__(
        self,
        cfg: DictConfig,
        data_path: Path | str,
        id2label: Optional[dict[int, str]] = None,
    ) -> None:
        """Initialize the InferDataModule.

        Args:
            cfg: Hydra configuration object.
            data_path: Path to directory with images or single image file.
            id2label: Optional mapping from class id to label name.
        """
        super().__init__()
        self.cfg = cfg
        self.data_path = Path(data_path)
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.image_size = cfg.data.image_size
        self.id2label = id2label or {}

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

        self.dataset: Optional[InferDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the inference dataset.

        Args:
            stage: Current stage (ignored for inference).
        """
        self.dataset = InferDataset(
            data_path=self.data_path,
            transform=self.transform,
            image_size=self.image_size,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction data loader.

        Returns:
            DataLoader for prediction.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def image_paths(self) -> list[Path]:
        """Get list of image paths in the dataset.

        Returns:
            List of image paths.
        """
        if self.dataset is None:
            return []
        return self.dataset.image_paths
