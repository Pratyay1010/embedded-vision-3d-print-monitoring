import os
import shutil
import random
import csv
import cv2
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ====================== 1. SETUP PATHS ======================
RAW_IMAGES_DIR = "data/images"
RAW_LABELS_DIR = "data/labels"
OUTPUT_DIR = "yolo_dataset_output"
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ====================== 2. SPLIT DATASET ======================
def split_dataset():
    # Get all image files
    image_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)

    train_files, temp_files = train_test_split(image_files, test_size=(1 - TRAIN_RATIO))
    val_files, test_files = train_test_split(temp_files, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO))

    os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/test", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/test", exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    def copy_files(files, split):
        for f in files:
            shutil.copy2(f"{RAW_IMAGES_DIR}/{f}", f"{OUTPUT_DIR}/images/{split}/{f}")
            label_file = f.replace('.jpg', '.txt').replace('.png', '.txt')
            shutil.copy2(f"{RAW_LABELS_DIR}/{label_file}", f"{OUTPUT_DIR}/labels/{split}/{label_file}")

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    print(f"Dataset split complete: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

# ====================== 3. CREATE YAML FILE ======================
def create_yaml():
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
test: images/test

names:
  0: cracking
  1: stringing
  2: warping
"""
    with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
        f.write(yaml_content.strip())
    print("YAML file created at:", f"{OUTPUT_DIR}/dataset.yaml")

# ====================== 4. TRAIN YOLOv8 AND SAVE MODEL ======================
def train_yolo():
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data=f"{OUTPUT_DIR}/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        optimizer="Adam",
        lr0=0.001,
        augment=True,
        project=OUTPUT_DIR,
        name="training",
        save=True,
        exist_ok=True
    )
    
    possible_locations = [
        os.path.join(OUTPUT_DIR, "training", "weights", "best.pt"),
        os.path.join("runs", "detect", "training", "weights", "best.pt"),
        os.path.join(OUTPUT_DIR, "train", "weights", "best.pt"),
        os.path.join("runs", "detect", "train", "weights", "best.pt")
    ]
    
    best_model_src = None
    for location in possible_locations:
        if os.path.exists(location):
            best_model_src = location
            break
    
    if not best_model_src:
        raise FileNotFoundError("Could not find trained model at any expected location")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_dst = os.path.join(MODEL_DIR, "best.pt")
    
    shutil.copy2(best_model_src, best_model_dst)
    print(f"Model successfully saved to: {os.path.abspath(best_model_dst)}")
    
    return model

# ====================== 5. RUN PREDICTIONS ======================
def run_predictions(model):
    metrics = model.val()
    
    metrics_data = [
        ["Metric", "Value"],
        ["Precision", metrics.box.mp],
        ["Recall", metrics.box.mr],
        ["mAP50", metrics.box.map50],
        ["mAP50-95", metrics.box.map],
        ["", ""],
        ["Class", "Precision", "Recall"],
        ["Cracking (0)", metrics.box.p[0], metrics.box.r[0]],
        ["Stringing (1)", metrics.box.p[1], metrics.box.r[1]],
        ["Warping (2)", metrics.box.p[2], metrics.box.r[2]]
    ]
    
    with open(os.path.join(PREDICTIONS_DIR, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics_data)
    
    font_scale = 1.5
    thickness = 3
    colors = {
        0: (0, 255, 0),   # Cracking - Green
        1: (255, 0, 0),    # Stringing - Blue
        2: (0, 0, 255)     # Warping - Red
    }
    
    test_images_dir = os.path.join(OUTPUT_DIR, "images/test")
    for img_name in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_name)
        img = cv2.imread(img_path)
        results = model.predict(img_path, save=False, conf=0.5)  # 50% confidence threshold
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls.item())
                label = f"{model.names[cls_id]}"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls_id], thickness)
                
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                cv2.rectangle(img, 
                            (x1, y1 - text_height - 15),
                            (x1 + text_width, y1 - 5),
                            colors[cls_id], -1)
                
                cv2.putText(img, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        cv2.imwrite(os.path.join(PREDICTIONS_DIR, f"pred_{img_name}"), img)
    
    print(f"Predictions saved to: {PREDICTIONS_DIR}")

# ====================== 6. RUN ALL STEPS ======================
if __name__ == "__main__":
    split_dataset()
    create_yaml()
    trained_model = train_yolo()
    run_predictions(trained_model)
    
    print("Pipeline executed successfully!")