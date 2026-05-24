import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

DATASET_PATH = "dataset"
CLASSES = ["stringing", "cracking", "warping"]

class SimpleDefectDataset(Dataset):
    def __init__(self, img_dir, label_dir, input_size=640):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.input_size = input_size
        self.images = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.transpose(2, 0, 1) / 255.0
        return torch.FloatTensor(img)

model = YOLO("yolo11n.pt")
model.train(data=f"{DATASET_PATH}/data.yaml", epochs=10, imgsz=640, batch=8)
metrics = model.val()
results = model.predict(f"{DATASET_PATH}/test", save=True)