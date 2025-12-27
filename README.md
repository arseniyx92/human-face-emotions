# Human Face Emotions Classification

Классификация эмоций на лицах людей с использованием PyTorch Lightning, Hydra и MLflow.

## 📋 Содержание

- [Установка](#установка)
- [Структура проекта](#структура-проекта)
- [Подготовка данных](#подготовка-данных)
- [Обучение](#обучение)
- [Инференс](#инференс)

## 🚀 Установка

### Требования

- Python 3.9+
- Poetry

### Шаги установки

```bash
# Клонирование репозитория
git clone <repository-url>
cd human-face-emotions

# Установка зависимостей
poetry install

# Установка pre-commit hooks
poetry run pre-commit install
```

### 📁 Структура проекта

```
human-face-emotions/
├── configs/                      # Hydra конфигурации
│   ├── config.yaml              # Основной конфиг
│   ├── model/
│   │   ├── cnn.yaml
│   │   └── cnn_large.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── fast.yaml
│   ├── data/
│   │   └── default.yaml
│   └── logging/
│       ├── mlflow.yaml
│       └── tensorboard.yaml
├── data/                         # Данные для обучения
│   ├── Angry/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   └── Surprised/
├── human_face_emotions/          # Исходный код
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── faces_dataset.py
│   │   ├── infer_dataset.py
│   │   └── model.py
│   ├── __init__.py
│   ├── train.py
│   └── infer.py
├── checkpoints/                  # Сохранённые модели
├── outputs/                      # Hydra outputs
├── mlruns/                       # MLflow данные
├── docs/                         # Sphinx документация
├── tests/                        # Тесты
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

### 📊 Подготовка данных

```
data/
├── Angry/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── Fear/
│   ├── 0.png
│   └── ...
├── Happy/
│   ├── 0.png
│   └── ...
├── Sad/
│   ├── 0.png
│   └── ...
└── Surprised/
    ├── 0.png
    └── ...
```
Поддерживаемые форматы: .png, .jpg, .jpeg

### 🎯 Обучение

#### Базовое обучение (GPU)
```bash
poetry run python -m human_face_emotions.train
```

#### Обучение на CPU
```bash
poetry run python -m human_face_emotions.train training.accelerator=cpu
```

#### Быстрое обучение (для тестирования)
```bash
poetry run python -m human_face_emotions.train training=testing_2_epochs
```

### 🔮 Инференс

#### Одно изображение
```bash
poetry run python -m human_face_emotions.infer \
    '+checkpoint_path="checkpoints/emotion-epoch=01-val_loss=1.27.ckpt"' \
    '+images_path="data/Happy/25.png"'
```

#### Директория с изображениями
```bash
poetry run python -m human_face_emotions.infer \
    '+checkpoint_path="checkpoints/emotion-epoch=01-val_loss=1.27.ckpt"' \
    '+images_path="./test_images/"'
```

#### Пример вывода

```bash
============================================================
Inference Configuration:
============================================================
Checkpoint: checkpoints/emotion-epoch=01-val_loss=1.27.ckpt
Images: data/Happy/25.png
Output: stdout
Accelerator: cpu
============================================================

Results:
{
    "model_checkpoint": "checkpoints/emotion-epoch=01-val_loss=1.27.ckpt",
    "images_path": "data/Happy/25.png",
    "num_images": 1,
    "predictions": [
        {
            "image_path": "data/Happy/25.png",
            "predicted_class": 2,
            "predicted_emotion": "Happy",
            "confidence": 0.8934,
            "all_probabilities": {
                "Angry": 0.0234,
                "Fear": 0.0123,
                "Happy": 0.8934,
                "Sad": 0.0456,
                "Surprised": 0.0253
            }
        }
    ]
}

============================================================
Summary:
============================================================
  Happy: 1
------------------------------------------------------------
  25.png: Happy (89.34%)
============================================================
```

#### Запуск MLflow UI
```bash
poetry run mlflow ui --backend-store-uri ./mlruns --port 5000
```