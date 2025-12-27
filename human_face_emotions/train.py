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
        # Устанавливаем tracking URI до создания логгера
        tracking_uri = cfg.logging.mlflow.tracking_uri

        # Преобразуем относительный путь в абсолютный
        if not tracking_uri.startswith("http"):
            tracking_uri = str(Path(tracking_uri).absolute())

        mlflow.set_tracking_uri(tracking_uri)

        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow experiment: {cfg.logging.mlflow.experiment_name}")

        logger = MLFlowLogger(
            experiment_name=cfg.logging.mlflow.experiment_name,
            tracking_uri=tracking_uri,
            run_name=cfg.logging.mlflow.run_name,
            tags=(
                OmegaConf.to_container(cfg.logging.mlflow.tags, resolve=True)
                if cfg.logging.mlflow.tags
                else None
            ),
            log_model=cfg.logging.mlflow.get("log_model", True),
        )

        print(f"MLflow run ID: {logger.run_id}")

        return logger

    elif cfg.logging.logger == "tensorboard":
        return TensorBoardLogger(
            save_dir=cfg.logging.tensorboard.save_dir,
            name=cfg.logging.tensorboard.name,
        )
    else:
        print("Warning: No logger configured")
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
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
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

    # Get logger BEFORE trainer
    logger = get_logger(cfg)

    if logger is None:
        print("WARNING: No logger initialized!")

    # Get callbacks
    callbacks = get_callbacks(cfg)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=logger,  # Убедитесь, что logger передаётся
        enable_progress_bar=cfg.logging.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.fit(model, datamodule)

    # Test
    print("\n" + "=" * 60)
    print("Running test...")
    print("=" * 60)
    trainer.test(model, datamodule)

    # Log artifacts to MLflow
    if logger and isinstance(logger, MLFlowLogger):
        print("\n" + "=" * 60)
        print("Logging artifacts to MLflow...")
        print("=" * 60)

        with mlflow.start_run(run_id=logger.run_id):
            # Log best checkpoint
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path and Path(best_model_path).exists():
                mlflow.log_artifact(best_model_path, "checkpoints")
                print(f"Logged checkpoint: {best_model_path}")

            # Export and log ONNX model
            onnx_path = Path("models/emotion_classifier.onnx")
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            model.export_to_onnx(onnx_path)
            mlflow.log_artifact(str(onnx_path), "onnx")
            print(f"Logged ONNX model: {onnx_path}")

            # Log labels
            labels_path = Path("models/labels.json")
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            labels_path.write_text(json.dumps(datamodule.id2label, indent=2))
            mlflow.log_artifact(str(labels_path), "config")
            print(f"Logged labels: {labels_path}")

            # Log Hydra config
            config_path = Path("models/config.yaml")
            config_path.write_text(OmegaConf.to_yaml(cfg))
            mlflow.log_artifact(str(config_path), "config")
            print(f"Logged config: {config_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")

    if logger and isinstance(logger, MLFlowLogger):
        print(f"MLflow run ID: {logger.run_id}")
        print(
            f"View in MLflow UI: mlflow ui --backend-store-uri {cfg.logging.mlflow.tracking_uri}"
        )


if __name__ == "__main__":
    main()
