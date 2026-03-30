import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLOWorld

class GroundingDinoDetector:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base", threshold=0.35):
        """
        Initializes the Grounding DINO zero-shot object detection pipeline.
        This is a much heavier, state-of-the-art text-to-bbox model.
        """
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"[GroundingDINO] Loading {model_id} on device: {'GPU' if self.device == 0 else 'CPU'}")
        
        self.detector = pipeline(
            task="zero-shot-object-detection",
            model=model_id,
            device=self.device
        )
        self.threshold = threshold
        # Grounding DINO works well with specific descriptive concepts to avoid confusing them with fences
        self.candidate_labels = ["metal u-shaped bike rack", "bicycle rack"]

    def detect(self, image: Image.Image):
        predictions = self.detector(
            image,
            candidate_labels=self.candidate_labels
        )
        # Filter by confidence threshold
        results = [p for p in predictions if p["score"] >= self.threshold]
        return results

class YoloWorldDetector:
    def __init__(self, model_id="yolov8s-world.pt", threshold=0.15):
        """
        Initializes the YOLO-World zero-shot model.
        It is exceptionally fast and optimized for edge detection.
        """
        print(f"[YOLO-World] Loading {model_id}...")
        self.model = YOLOWorld(model_id)
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
