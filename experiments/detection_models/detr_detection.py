import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from pathlib import Path
import cv2
import numpy as np

DATASET_PATH = "dataset"

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.config.num_labels = 3
model.config.id2label = {0: "stringing", 1: "cracking", 2: "warping"}

class DetrDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = Path(img_dir)
        self.images = list(self.img_dir.glob("*.jpg"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = processor(images=img, return_tensors="pt")
        return {k: v.squeeze() for k, v in encoding.items()}

dataset = DetrDataset(f"{DATASET_PATH}/test")
dataloader = DataLoader(dataset, batch_size=4)

model.eval()
with torch.no_grad():
    for batch in dataloader:
        outputs = model(pixel_values=batch["pixel_values"])
        logits, pred_boxes = outputs.logits, outputs.pred_boxes
        print(f"Predictions: {logits.shape}")
        break