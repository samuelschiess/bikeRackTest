import torch
from PIL import Image
from ultralytics import YOLOWorld

class YoloWorldDetector:
    def __init__(self, model_id="yolov8s-world.pt", threshold=0.15):
        """
        Initializes the YOLO-World zero-shot model.
        It is exceptionally fast and optimized for edge detection.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLO-World] Loading {model_id} on {self.device.upper()}...")
        
        self.model = YOLOWorld(model_id)
        self.model.to(self.device)
        
        # YOLO-World lets us set arbitrary classes!
        self.model.set_classes(["bike rack", "bicycle rack", "bicycle parking"])
        self.threshold = threshold

    def detect(self, image: Image.Image):
        # image is a PIL Image
        results = self.model(image, conf=self.threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                score = box.conf[0].item()
                label = self.model.names[int(box.cls[0])]
                detections.append({
                    "score": score,
                    "label": label,
                    "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
                })
        return detections
