"""Training script with Hydra and MLflow for the emotion classifier."""

import json
from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from human_face_emotions.modules import EmotionClassifierModule, FacesEmotionDataModule


def get_logger(cfg: DictConfig):
    """Create appropriate logger based on configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        PyTorch Lightning logger instance.
    """
    if cfg.logging.logger == "mlflow":
        return MLFlowLogger(
            experiment_name=cfg.logging.mlflow.experiment_name,
            tracking_uri=cfg.logging.mlflow.tracking_uri,
            run_name=cfg.logging.mlflow.run_name,
            tags=OmegaConf.to_container(cfg.logging.mlflow.tags, resolve=True),
            log_model=cfg.logging.mlflow.log_model,
        )
    elif cfg.logging.logger == "tensorboard":
        return TensorBoardLogger(
            save_dir=cfg.logging.tensorboard.save_dir,
            name=cfg.logging.tensorboard.name,
        )
    else:
        return None


def get_callbacks(cfg: DictConfig) -> list:
    """Create training callbacks based on configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of PyTorch Lightning callbacks.
    """
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.training.checkpoint.monitor,
            dirpath="checkpoints",
            filename="emotion-{epoch:02d}-{val_loss:.2f}",
            save_top_k=cfg.training.checkpoint.save_top_k,
            mode=cfg.training.checkpoint.mode,
        ),
        EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return callbacks


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train emotion classifier with Hydra configuration.

    Args:
        cfg: Hydra configuration object (automatically injected).
    """
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Initialize datamodule
    datamodule = FacesEmotionDataModule(cfg)
    datamodule.setup()

    print(f"Number of classes: {datamodule.num_classes}")
    print(f"Classes: {datamodule.id2label}")

    # Update config with actual number of classes
    cfg.model.num_classes = datamodule.num_classes

    # Initialize model
    model = EmotionClassifierModule(cfg=cfg, id2label=datamodule.id2label)

    # Get logger
    logger = get_logger(cfg)

    # Log hyperparameters
    if logger and isinstance(logger, MLFlowLogger):
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Get callbacks
    callbacks = get_callbacks(cfg)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=cfg.logging.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, datamodule)

    # Test
    print("Running test...")
    trainer.test(model, datamodule)

    # Log artifacts
    if mlflow.active_run():
        # Log best checkpoint
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            mlflow.log_artifact(best_model_path, "checkpoints")

        # Export and log ONNX model
        onnx_path = Path("models/emotion_classifier.onnx")
        model.export_to_onnx(onnx_path)
        mlflow.log_artifact(str(onnx_path), "onnx")

        # Log labels
        labels_path = Path("models/labels.json")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(datamodule.id2label, indent=2))
        mlflow.log_artifact(str(labels_path), "config")

        # Log Hydra config
        config_path = Path("models/config.yaml")
        config_path.write_text(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(str(config_path), "config")

    print(f"\nTraining complete!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
