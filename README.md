# Детекция малярии с использованием Faster R-CNN

## 📦 Описание
Курсовой проект: автоматическое определение заражённых клеток малярии по микроскопическим изображениям. Используется модель Faster R-CNN из библиотеки Detectron2.

## 🧠 Участники
- Санников Максим Витальевич
- Смирнов Игорь Сергеевич

## 🚀 Запуск в Google Colab
1. Перейдите в [Google Colab](https://colab.research.google.com)
2. Загрузите `МКРТ.ipynb`
3. Загрузите свой `kaggle.json`, как описано ниже

## 📥 Подготовка Kaggle API
1. Перейдите в [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Найдите раздел **API** и нажмите **Create New API Token**
3. Скачайте `kaggle.json`
4. В Google Colab загрузите его:

```python
from google.colab import files
files.upload()  # Загрузите kaggle.json
```

## 💾 Установка зависимостей

```bash
pip install -U pip
pip install kaggle
pip install -U torch torchvision
pip install git+https://github.com/facebookresearch/detectron2.git
```

## 🖥 Локальный запуск (например, PyCharm)

```bash
python train_model.py
```

## 📊 Метрики

| Модель           | mAP@[.5:.95] | Recall | RBC  | Trophozoite | Остальные |
|------------------|-------------|--------|------|--------------|------------|
| Faster R-CNN (2500 ит.) | 19.3%       | 23%    | 73.9% | 42.2%        | 0%         |
| Faster R-CNN (5000 ит.) | 20.1%       | 25%    | 75.1% | 45.3%        | 0%         |

## 📁 Структура
```
├── МКРТ.ipynb            # ноутбук для Google Colab
├── train_model.py        # запуск модели локально
├── requirements.txt      # зависимости
└── README.md             # инструкция
```
