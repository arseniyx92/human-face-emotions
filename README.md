# Human Face Emotions Classification

Классификация эмоций на лицах людей с использованием PyTorch Lightning, Hydra и MLflow.

## Содержание

- [О проекте](#о-проекте)
- [Постановка задачи](#постановка-задачи)
- [Архитектура решения](#архитектура-решения)
- [Установка](#установка)
- [Структура проекта](#структура-проекта)
- [Подготовка данных](#подготовка-данных)
- [Обучение](#обучение)
- [Инференс](#инференс)
- [Метрики и качество](#метрики-и-качество)

## О проекте

**Human Face Emotions Classification** — система для автоматического распознавания эмоций по изображениям человеческих лиц. Проект решает задачу многоклассовой классификации на 5 категорий эмоций: **Angry, Fear, Happy, Sad, Surprised**.

**Автор:** Хлытчиев Арсений Денисович

**Ценность решения:**
- Интерактивные приложения (фильтры AR, игры)
- Системы анализа вовлеченности пользователей
- Исследования в психологии и нейробиологии
- Системы безопасности и мониторинга

## Постановка задачи

**Сухой остаток:** Многоклассовая классификация эмоций по изображениям человеческих лиц на 5 категорий.

**Основные сложности:**
- Высокая вариативность изображений (освещение, ракурс, возраст, этническая принадлежность)
- Некоторые изображения содержат артефакты, водяные знаки или неявно выраженные эмоции
- Необходимость устойчивого обобщения на разнородных данных

**Технические требования:**
- Input: Изображение лица в формате JPEG/PNG
- Output: JSON с предсказанной эмоцией и вероятностями для всех классов
- Целевые метрики: Accuracy > 0.88, Weighted F1-Score > 0.87

## Архитектура решения

### Бейзлайн модель
- Простая CNN (3-4 сверточных блока + полносвязный классификатор)
- Минимальная аугментация данных
- Цель: воспроизвести accuracy ~0.89 как точку отсчета

### Основная модель
- **Архитектура:** Глубокая CNN на основе EfficientNet-B0/ResNet-18 с fine-tuning
- **Аугментация данных:**
  - RandomHorizontalFlip
  - RandomRotation (±10°)
  - ColorJitter (яркость, контраст, насыщенность)
  - RandomAffine, RandomPerspective
  - Нормализация (статистики ImageNet)
- **Классификатор:** Global Average Pooling + FC с Dropout (0.4-0.5)
- **Loss:** CrossEntropyLoss
- **Optimizer:** AdamW с CosineAnnealingLR

### Дополнительная сложность (опционально)
- Self-Supervised Pretext Task (Masked Autoencoding)
- Vision Transformer (ViT-tiny) в качестве энкодера
- Предварительное обучение на датасете лиц без меток

## Установка

### Требования

- Python 3.11+
- Poetry для управления зависимостями
- GPU (рекомендуется) для ускорения обучения

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

## Структура проекта

```bash
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

## Подготовка данных

```bash
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

## Обучение

#### Базовое обучение (GPU)
```bash
poetry run python -m human_face_emotions.train
```

### Обучение на CPU
```bash
poetry run python -m human_face_emotions.train training.accelerator=cpu
```

### Быстрое обучение (для тестирования)
```bash
poetry run python -m human_face_emotions.train training=testing_2_epochs
```

## Инференс

### Одно изображение
```bash
poetry run python -m human_face_emotions.infer \
    '+checkpoint_path="checkpoints/emotion-epoch=01-val_loss=1.27.ckpt"' \
    '+images_path="data/Happy/25.png"'
```

### Директория с изображениями
```bash
poetry run python -m human_face_emotions.infer \
    '+checkpoint_path="checkpoints/emotion-epoch=01-val_loss=1.27.ckpt"' \
    '+images_path="./test_images/"'
```

### Пример вывода

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

### Запуск MLflow UI
```bash
poetry run mlflow ui --backend-store-uri ./mlruns --port 5000
```

## Метрики и качество

### Целевые метрики
| Метрика | Цель | Описание |
|---------|------|----------|
| **Accuracy** | **>0.88** | Основная метрика классификации |
| **Weighted F1** | **>0.87** | Баланс precision/recall |
| **Per-Class Recall** | **>0.80** | Качество по каждому классу |

### Baseline (Kaggle)
- **Простая CNN:** Accuracy ≈ 0.8955
- **Наша цель:** Превысить 0.90 с улучшенной архитектурой

### Воспроизводимость
Train/Val/Test: 64%/16%/20% (стратифицировано)

Random Seed: 281 (воспроизводимость)

Ранняя остановка: по val_loss (patience=10)

Логирование: MLflow для отслеживания экспериментов