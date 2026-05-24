import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np

DATASET_PATH = "dataset"
NUM_CLASSES = 3
IMG_SIZE = 224

class DefectClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier[-1] = nn.Linear(1024, NUM_CLASSES)
    
    def forward(self, x):
        return self.backbone(x)

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.classes = ["stringing", "cracking", "warping"]
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*.*"):
                self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.transpose(2, 0, 1) / 255.0
        return torch.FloatTensor(img), torch.tensor(label)

model = DefectClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = ImageDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(5):
    for imgs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

torch.save(model.state_dict(), "mobilenetv3_defect.pth")