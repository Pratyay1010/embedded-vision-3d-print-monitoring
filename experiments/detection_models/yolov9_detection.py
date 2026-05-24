from ultralytics import YOLO
import torch

DATASET_PATH = "dataset"

model = YOLO("yolov9n.pt")

model.train(
    data=f"{DATASET_PATH}/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=0
)

model.export(format="onnx", imgsz=640)

results = model(f"{DATASET_PATH}/test", stream=True)
for r in results:
    boxes = r.boxes
    print(f"Detected {len(boxes)} defects")