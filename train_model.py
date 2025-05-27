import os
import json
from tqdm import tqdm
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

def download_dataset_if_needed():
    if not os.path.exists("malaria-bounding-boxes.zip"):
        print("📦 Скачиваем датасет с Kaggle...")
        os.system("mkdir -p ~/.kaggle")
        if not os.path.exists("kaggle.json"):
            raise FileNotFoundError("❌ Файл kaggle.json не найден! Скачайте его с https://www.kaggle.com/account и положите в корень проекта.")
        os.system("cp kaggle.json ~/.kaggle/kaggle.json")
        os.system("chmod 600 ~/.kaggle/kaggle.json")
        os.system("kaggle datasets download -d kmader/malaria-bounding-boxes")
    else:
        print("✅ Датасет уже загружен.")

    if not os.path.exists("malaria"):
        print("📂 Распаковываем архив...")
        os.system("unzip -qo malaria-bounding-boxes.zip -d malaria")
    else:
        print("✅ Архив уже распакован.")

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

    print(f"✅ Сохранено: {output_path}")

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

if __name__ == "__main__":
    train()
