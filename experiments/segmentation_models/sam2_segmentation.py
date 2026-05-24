from ultralytics import SAM
from pathlib import Path
import cv2
import numpy as np

DATASET_PATH = "dataset"

model = SAM("sam_b.pt")
model.info()

test_images = list(Path(DATASET_PATH).glob("*.jpg")) + list(Path(DATASET_PATH).glob("*.png"))

for img_path in test_images[:5]:
    print(f"\nProcessing: {img_path.name}")
    
    results_bbox = model(str(img_path), bboxes=[100, 100, 300, 300])
    if results_bbox[0].masks is not None:
        print(f"  Bbox prompt: {len(results_bbox[0].masks)} masks")
    
    h, w = cv2.imread(str(img_path)).shape[:2]
    center_point = [w//2, h//2]
    results_point = model(str(img_path), points=[center_point], labels=[1])
    if results_point[0].masks is not None:
        print(f"  Single point prompt: mask shape {results_point[0].masks.shape}")
    
    results_multi = model(str(img_path), points=[[w//3, h//2], [2*w//3, h//2]], labels=[1, 1])
    if results_multi[0].masks is not None:
        print(f"  Multi-point prompt: generated segmentation")
    
    results_neg = model(str(img_path), points=[[[w//2, h//2], [w//4, h//4]]], labels=[[1, 0]])
    if results_neg[0].masks is not None:
        mask_data = results_neg[0].masks.data[0].cpu().numpy()
        mask_vis = (mask_data * 255).astype(np.uint8)
        cv2.imwrite(f"sam2_{img_path.stem}.png", mask_vis)

for result in results_multi:
    if hasattr(result, 'boxes') and result.boxes is not None:
        print(f"Boxes: {result.boxes}")
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        print(f"Keypoints: {result.keypoints}")