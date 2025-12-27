"""Model module for Human Face Emotions classification.

This module provides PyTorch Lightning module wrapping a CNN model
for facial emotion classification.
"""

from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig


class EmotionCNN(nn.Module):
    """Convolutional Neural Network for facial emotion classification.

    Architecture consists of configurable convolutional blocks followed by
    a fully connected classification head.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_size: int = 48,
        dropout: float = 0.5,
        conv_channels: list[int] = None,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        """Initialize the EmotionCNN.

        Args:
            num_classes: Number of emotion classes to predict. Defaults to 5.
            input_size: Input image size (assumed square). Defaults to 48.
            dropout: Dropout rate for regularization. Defaults to 0.5.
            conv_channels: List of channel sizes for conv layers. Defaults to [32, 64, 128].
            kernel_size: Kernel size for conv layers. Defaults to 3.
            padding: Padding for conv layers. Defaults to 1.
        """
        super().__init__()

        conv_channels = conv_channels or [32, 64, 128]
        in_channels = 3

        conv_layers = []
        for out_channels in conv_channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Calculate flattened size after conv layers
        num_pools = len(conv_channels)
        final_size = input_size // (2**num_pools)
        flattened_size = final_size * final_size * conv_channels[-1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flattened_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        x = self.conv(x)
        x = self.head(x)
        return x


class EmotionClassifierModule(pl.LightningModule):
    """PyTorch Lightning module for emotion classification.

    Handles training, validation, testing, and inference.
    """

    def __init__(
        self,
        cfg: DictConfig,
        id2label: Optional[dict[int, str]] = None,
    ) -> None:
        """Initialize the EmotionClassifierModule.

        Args:
            cfg: Hydra configuration object.
            id2label: Optional mapping from class id to label name.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["id2label"])

        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        self.id2label = id2label or {i: str(i) for i in range(self.num_classes)}

        # Build model
        self.model = EmotionCNN(
            num_classes=cfg.model.num_classes,
            input_size=cfg.model.input_size,
            dropout=cfg.model.dropout,
            conv_channels=list(cfg.model.conv_channels),
            kernel_size=cfg.model.kernel_size,
            padding=cfg.model.padding,
        )

        self.criterion = nn.CrossEntropyLoss()

        # Training metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="weighted"
        )

        # Validation metrics
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="weighted"
        )

        # Test metrics
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_classes, average="weighted"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        return self.model(x)

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Common step for training, validation, and testing.

        Args:
            batch: Tuple of (images, labels).

        Returns:
            Tuple of (loss, predictions, labels).
        """
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.

        Returns:
            Training loss.
        """
        loss, preds, labels = self._common_step(batch)

        self.train_accuracy.update(preds, labels)
        self.train_f1.update(preds, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_f1", self.train_f1, prog_bar=False, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single validation step.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.

        Returns:
            Validation loss.
        """
        loss, preds, labels = self._common_step(batch)

        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_acc", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("val_f1", self.val_f1, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single test step.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.

        Returns:
            Test loss.
        """
        loss, preds, labels = self._common_step(batch)

        self.test_accuracy.update(preds, labels)
        self.test_f1.update(preds, labels)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_acc", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("test_f1", self.test_f1, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Perform a single prediction step.

        Args:
            batch: Dictionary with 'pixel_values' and 'image_path'.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary with prediction results.
        """
        pixel_values = batch["pixel_values"]
        image_paths = batch["image_path"]

        logits = self(pixel_values)
        probabilities = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(probabilities, dim=1)
        confidences = probabilities.gather(1, pred_classes.unsqueeze(1)).squeeze(1)

        return {
            "image_paths": image_paths,
            "class_ids": pred_classes,
            "class_names": [self.id2label[idx.item()] for idx in pred_classes],
            "confidences": confidences,
            "all_probabilities": probabilities,
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        scheduler_cfg = self.cfg.training.scheduler
        if scheduler_cfg.name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_cfg.factor,
                patience=scheduler_cfg.patience,
                # verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def export_to_onnx(self, output_path: Path) -> None:
        """Export the model to ONNX format.

        Args:
            output_path: Path to save the ONNX model.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        input_size = self.cfg.model.input_size
        dummy_input = torch.randn(1, 3, input_size, input_size)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=16,
        )
