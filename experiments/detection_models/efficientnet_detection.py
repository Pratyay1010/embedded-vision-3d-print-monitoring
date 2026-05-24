import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path

DATASET_PATH = "dataset"
IMG_SIZE = 256

class EfficientNetDefect(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, 3)
    
    def forward(self, x):
        return self.model(x)

class DefectDataset(Dataset):
    def __init__(self, split="train"):
        self.path = Path(DATASET_PATH) / split
        self.images = list(self.path.rglob("*.jpg")) + list(self.path.rglob("*.png"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.images[idx]))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(0)
        return img, label

model = EfficientNetDefect()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
train_loader = DataLoader(DefectDataset("train"), batch_size=16, shuffle=True)

model.train()
for epoch in range(8):
    total_loss = 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        preds = model(imgs)
        loss = nn.CrossEntropyLoss()(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}")