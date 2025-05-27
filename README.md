# Детекция малярии с использованием Faster R-CNN

## 🧠 Участники
- Санников Максим Витальевич
- Смирнов Игорь Сергеевич

---

## 📦 Описание проекта
Автоматическое обнаружение заражённых клеток малярии с помощью модели **Faster R-CNN** из библиотеки Detectron2.

---

## 🚀 Быстрый старт

### Способ 1: Google Colab (рекомендуется)
1. Перейдите в [Google Colab](https://colab.research.google.com)
2. Загрузите файл `МКРТ.ipynb`
3. Загрузите свой `kaggle.json` (см. ниже)
4. Запустите ноутбук — он автоматически:
   - скачает датасет
   - конвертирует его в COCO формат
   - обучит модель Faster R-CNN
   - отобразит метрики и визуализацию

---

### 💻 Способ 2: Локальный запуск

#### 1. Установите базовые зависимости:
```bash
pip install -r requirements_base.txt
```

#### 2. Установите Detectron2:

На **Windows**:
```bash
install_detectron2.bat
```

На **Linux / macOS**:
```bash
bash install_detectron2.sh
```

#### 3. Скачайте `kaggle.json`:
   - Перейдите в [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
   - Нажмите "Create New API Token"
   - Скачайте файл и поместите его в корень проекта

#### 4. Запустите:
```bash
python train_model.py
```

Скрипт:
- скачает датасет `malaria-bounding-boxes` с Kaggle
- распакует его
- обучит модель
- выведет метрики
- отобразит визуализацию

---

## 📥 Где взять датасет?

Датасет берётся отсюда:  
🔗 https://www.kaggle.com/datasets/kmader/malaria-bounding-boxes

**Но вручную скачивать не нужно** — скрипт и ноутбук делают это автоматически через `kaggle.json`.

---

## 📁 Структура проекта

```
malaria_project/
├── kaggle.json
├── train_model.py
├── МКРТ.ipynb
├── requirements_base.txt
├── install_detectron2.bat
├── install_detectron2.sh
└── README.md
```

---

## 📊 Метрики

| Модель           | mAP@[.5:.95] | Recall | RBC  | Trophozoite | Остальные |
|------------------|-------------|--------|------|--------------|------------|
| Faster R-CNN (2500 ит.) | 19.3%       | 23%    | 73.9% | 42.2%        | 0%         |
| Faster R-CNN (5000 ит.) | 20.1%       | 25%    | 75.1% | 45.3%        | 0%         |

---

## ⚠️ Примечания
- Модель работает хорошо для классов с большим числом примеров (`red blood cell`, `trophozoite`)
- Редкие классы не распознаются — требуется доработка (focal loss, аугментации)


