"""Inference script with Hydra for Human Face Emotions classification."""

import json
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.serialization
from omegaconf import DictConfig, ListConfig, OmegaConf

from human_face_emotions.modules import EmotionClassifierModule, InferDataModule


# Default label mapping
DEFAULT_ID2LABEL = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprised",
}

# Разрешаем OmegaConf классы для загрузки checkpoint
torch.serialization.add_safe_globals([DictConfig, ListConfig])


def load_model(
    checkpoint_path: Path | str,
    cfg: DictConfig,
    id2label: Optional[dict[int, str]] = None,
) -> EmotionClassifierModule:
    """Load a trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    id2label = id2label or DEFAULT_ID2LABEL

    # Загружаем checkpoint вручную с weights_only=False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = EmotionClassifierModule(cfg=cfg, id2label=id2label)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def run_inference(
    model: EmotionClassifierModule,
    datamodule: InferDataModule,
    cfg: DictConfig,
) -> list[dict]:
    """Run inference on images.

    Args:
        model: Trained EmotionClassifierModule.
        datamodule: InferDataModule with images to process.
        cfg: Hydra configuration object.

    Returns:
        List of prediction results.
    """
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        enable_progress_bar=cfg.logging.enable_progress_bar,
        logger=False,
    )

    predictions = trainer.predict(model, datamodule)

    # Flatten results
    results = []
    for batch_pred in predictions:
        for i in range(len(batch_pred["image_paths"])):
            results.append(
                {
                    "image_path": batch_pred["image_paths"][i],
                    "predicted_class": batch_pred["class_ids"][i].item(),
                    "predicted_emotion": batch_pred["class_names"][i],
                    "confidence": round(batch_pred["confidences"][i].item(), 4),
                    "all_probabilities": {
                        model.id2label[j]: round(prob.item(), 4)
                        for j, prob in enumerate(batch_pred["all_probabilities"][i])
                    },
                }
            )

    return results


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run inference with Hydra configuration.

    Args:
        cfg: Hydra configuration object (automatically injected).

    Usage:
        python -m human_face_emotions.infer \
            '+checkpoint_path="checkpoints/best.ckpt"' \
            '+images_path="./test_images/"'
    """
    checkpoint_path = cfg.get("checkpoint_path")
    images_path = cfg.get("images_path")
    output_path = cfg.get("output_path")

    if not checkpoint_path:
        raise ValueError(
            "checkpoint_path must be specified. "
            "Usage: python -m human_face_emotions.infer '+checkpoint_path=\"path\"' '+images_path=\"path\"'"
        )
    if not images_path:
        raise ValueError(
            "images_path must be specified. "
            "Usage: python -m human_face_emotions.infer '+checkpoint_path=\"path\"' '+images_path=\"path\"'"
        )

    print("=" * 60)
    print("Inference Configuration:")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images: {images_path}")
    print(f"Output: {output_path or 'stdout'}")
    print(f"Accelerator: {cfg.training.accelerator}")
    print("=" * 60)

    # Set precision
    torch.set_float32_matmul_precision("medium")

    # Load labels if available
    checkpoint_dir = Path(checkpoint_path).parent
    labels_path = checkpoint_dir / "labels.json"

    if labels_path.exists():
        print(f"Loading labels from: {labels_path}")
        id2label = json.loads(labels_path.read_text())
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        print("Using default labels")
        id2label = DEFAULT_ID2LABEL

    print(f"Labels: {id2label}")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, cfg, id2label)
    print("Model loaded successfully!")

    # Setup datamodule
    print(f"\nLoading images from: {images_path}")
    datamodule = InferDataModule(cfg=cfg, data_path=images_path, id2label=id2label)
    datamodule.setup()
    print(f"Found {len(datamodule.image_paths)} images")

    # Run inference
    print("\nRunning inference...")
    results = run_inference(model, datamodule, cfg)

    # Format output
    output = {
        "model_checkpoint": str(checkpoint_path),
        "images_path": str(images_path),
        "num_images": len(results),
        "predictions": results,
    }

    formatted_output = json.dumps(output, indent=4, ensure_ascii=False)

    # Save or print results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(formatted_output)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nResults:")
        print(formatted_output)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    class_counts = {}
    for pred in results:
        emotion = pred["predicted_emotion"]
        class_counts[emotion] = class_counts.get(emotion, 0) + 1

    for emotion, count in sorted(class_counts.items()):
        print(f"  {emotion}: {count}")

    print("-" * 60)

    for pred in results:
        filename = Path(pred["image_path"]).name
        emotion = pred["predicted_emotion"]
        confidence = pred["confidence"]
        print(f"  {filename}: {emotion} ({confidence:.2%})")

    print("=" * 60)


if __name__ == "__main__":
    main()
