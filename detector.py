import torch
from PIL import Image
from ultralytics import YOLO, YOLOWorld

class YoloWorldDetector:
    def __init__(self, model_id="runs/detect/slc_bike_rack_model/weights/best.pt", threshold=0.05):
        """
        Initializes the YOLO-World zero-shot model OR a locally fine-tuned YOLOv8 model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLO Edge] Loading {model_id} on {self.device.upper()}...")
        
        # If loading a custom trained 'best.pt', bypass the zero-shot wrapper
        if "world.pt" not in model_id:
            self.model = YOLO(model_id)
        else:
            self.model = YOLOWorld(model_id)
            self.model.set_classes(["bike rack", "bicycle rack", "bicycle parking"])
            
        self.model.to(self.device)
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
