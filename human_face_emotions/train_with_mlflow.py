"""Training script with MLflow logging for the emotion classifier."""

from pathlib import Path
from typing import Optional

import fire
import mlflow
import pytorch_lightning as pl

from modules.faces_dataset import FacesEmotionDataModule
from modules.model import EmotionClassifierModule, get_mlflow_logger


def main(
    data_dir: str = "./data",
    max_epochs: int = 300,
    batch_size: int = 512,
    lr: float = 0.003,
    weight_decay: float = 0.01,
    dropout: float = 0.5,
    accelerator: str = "cpu",
    devices: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    experiment_name: str = "emotion-classification",
    run_name: Optional[str] = None,
    tracking_uri: str = "mlruns",
    export_onnx: bool = True,
) -> None:
    """Train emotion classifier with MLflow logging.

    Args:
        data_dir: Path to the data directory.
        max_epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        dropout: Dropout rate.
        accelerator: Accelerator type ('cpu', 'gpu', 'auto').
        devices: Number of devices to use.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducibility.
        experiment_name: MLflow experiment name.
        run_name: Optional MLflow run name.
        tracking_uri: MLflow tracking URI.
        export_onnx: Whether to export model to ONNX.

    Example:
        # Basic training with CPU
        python train_with_mlflow.py --data_dir=./data --accelerator=cpu

        # Training with GPU and custom experiment
        python train_with_mlflow.py --data_dir=./data --accelerator=gpu --experiment_name=my-experiment

        # With remote MLflow server
        python train_with_mlflow.py --data_dir=./data --tracking_uri=http://localhost:5000
    """
    pl.seed_everything(seed)

    # Initialize datamodule
    datamodule = FacesEmotionDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        cache_data=True,
    )
    datamodule.setup()

    print(f"Number of classes: {datamodule.num_classes}")
    print(f"Classes: {datamodule.id2label}")

    # Initialize model
    model = EmotionClassifierModule(
        num_classes=datamodule.num_classes,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        id2label=datamodule.id2label,
    )

    # MLflow logger
    mlflow_logger = get_mlflow_logger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags={
            "model": "EmotionCNN",
            "framework": "pytorch-lightning",
        },
    )

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="emotion-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=mlflow_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, datamodule)

    # Test
    print("Running test...")
    trainer.test(model, datamodule)

    # Log model artifact to MLflow
    if mlflow.active_run():
        # Log the best checkpoint
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            mlflow.log_artifact(best_model_path, "checkpoints")

        # Export and log ONNX model
        if export_onnx:
            onnx_path = Path("models/emotion_classifier.onnx")
            model.export_to_onnx(onnx_path)
            mlflow.log_artifact(str(onnx_path), "onnx")
            print(f"ONNX model exported to: {onnx_path}")

        # Log id2label mapping
        import json

        labels_path = Path("models/labels.json")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(datamodule.id2label, indent=2))
        mlflow.log_artifact(str(labels_path), "config")

    print(f"\nTraining complete!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"\nTo view MLflow UI, run: mlflow ui --backend-store-uri {tracking_uri}")


if __name__ == "__main__":
    fire.Fire(main)
