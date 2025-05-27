import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt
import random

# ✅ Настройка Kaggle API (кросс-платформенно)
def setup_kaggle():
    kaggle_json = Path("kaggle.json")
    if not kaggle_json.exists():
        raise FileNotFoundError("❌ Файл kaggle.json не найден! Скачайте его с https://www.kaggle.com/account и поместите в корень проекта.")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(kaggle_json), str(kaggle_dir / "kaggle.json"))
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    print("✅ Kaggle API настроен.")

# 📥 Загрузка датасета
def download_dataset_if_needed():
    if not os.path.exists("malaria-bounding-boxes.zip"):
        print("📦 Скачиваем датасет с Kaggle...")
        setup_kaggle()
        os.system("kaggle datasets download -d kmader/malaria-bounding-boxes")
    else:
        print("✅ Архив с датасетом уже скачан.")

    if not os.path.exists("malaria"):
        print("📂 Распаковываем архив...")
        os.system("unzip -qo malaria-bounding-boxes.zip -d malaria")
    else:
        print("✅ Архив уже распакован.")

# 🔄 Конвертация аннотаций в формат COCO
def convert_to_coco(source_path, image_dir, output_path):
    with open(source_path, 'r') as f:
        raw_data = json.load(f)

    coco_images = []
    coco_annotations = []
    coco_categories = {}
    category_id_counter = 1
    annotation_id = 1
    image_id = 1

    for item in tqdm(raw_data, desc="Конвертация в COCO"):
        image_info = item["image"]
        file_name = os.path.basename(image_info["pathname"])
        width = image_info["shape"]["c"]
        height = image_info["shape"]["r"]

        coco_images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        for obj in item["objects"]:
            category_name = obj["category"]
            if category_name not in coco_categories:
                coco_categories[category_name] = category_id_counter
                category_id_counter += 1

            cat_id = coco_categories[category_name]
            bbox = obj["bounding_box"]
            xmin = bbox["minimum"]["c"]
            ymin = bbox["minimum"]["r"]
            xmax = bbox["maximum"]["c"]
            ymax = bbox["maximum"]["r"]
            width_box = xmax - xmin
            height_box = ymax - ymin

            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    coco_format = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": i, "name": name} for name, i in coco_categories.items()
        ]
    }

    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"✅ COCO файл сохранён: {output_path}")

# 📊 Оценка модели
def evaluate_model(cfg):
    print("\n📊 Оценка модели на тестовом наборе...")
    evaluator = COCOEvaluator("malaria_test", cfg, False, output_dir="./output_malaria")
    val_loader = build_detection_test_loader(cfg, "malaria_test")
    model = DefaultPredictor(cfg).model
    results = inference_on_dataset(model, val_loader, evaluator)
    print("✅ Метрики:")
    print(results)

# 🖼️ Визуализация случайного изображения
def visualize_random_prediction(cfg):
    print("\n🎯 Визуализация случайного изображения:")
    predictor = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, "malaria_test")
    dataset = val_loader.dataset

    sample = random.choice(dataset)
    img_path = sample["file_name"]
    image = cv2.imread(img_path)
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("malaria_test"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(12, 12))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.title(os.path.basename(img_path))
    plt.show()

# 🖼️ Визуализация по конкретному пути
def visualize_specific_image(cfg, path):
    print(f"🎯 Визуализация конкретного изображения: {path}")
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(path)
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("malaria_test"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(12, 12))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.title(os.path.basename(path))
    plt.show()

# 🚀 Основной процесс обучения
def train():
    download_dataset_if_needed()

    base_path = "malaria/malaria"
    image_dir = os.path.join(base_path, "images")
    train_json = os.path.join(base_path, "malaria", "training.json")
    test_json = os.path.join(base_path, "malaria", "test.json")
    train_coco = os.path.join(base_path, "train_coco.json")
    test_coco = os.path.join(base_path, "test_coco.json")

    convert_to_coco(train_json, image_dir, train_coco)
    convert_to_coco(test_json, image_dir, test_coco)

    register_coco_instances("malaria_train", {}, train_coco, image_dir)
    register_coco_instances("malaria_test", {}, test_coco, image_dir)

    metadata = MetadataCatalog.get("malaria_train")
    dataset_dicts = DatasetCatalog.get("malaria_train")
    print(f"📦 Зарегистрировано {len(dataset_dicts)} изображений")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
    cfg.DATASETS.TRAIN = ("malaria_train",)
    cfg.DATASETS.TEST = ("malaria_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.OUTPUT_DIR = "output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluate_model(cfg)
    visualize_random_prediction(cfg)
    # visualize_specific_image(cfg, "malaria/malaria/images/your_image.png")  # Раскомментируй при необходимости

if __name__ == "__main__":
    train()
