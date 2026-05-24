import cv2
from ultralytics import YOLO

def run_camera_inference(model_path, conf_threshold=0.5):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    colors = {
        0: (0, 255, 0),   # Cracking - Green
        1: (255, 0, 0),   # Stringing - Blue
        2: (0, 0, 255)    # Warping - Red
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model.predict(frame, conf=conf_threshold)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                label = f"{result.names[cls_id]} {conf:.2f}"

                color = colors.get(cls_id, (255, 255, 255))  # Default to white if unknown class
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("YOLO Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_inference(
        model_path="models/best.pt"
    )