import pandas as pd
from datetime import datetime

class BikeRackDataStore:
    def __init__(self):
        self.records = []

    def add_record(self, image_id, model_name, lat, lon, captured_at, confidence, bbox, image_path=None):
        """
        Appends a detected bike rack to the in-memory store.
        """
        self.records.append({
            "image_id": image_id,
            "model_name": model_name,
            "image_path": image_path,
            "latitude": lat,
            "longitude": lon,
            "captured_at": captured_at,
            "confidence": confidence,
            "bbox_xmin": bbox["xmin"],
            "bbox_ymin": bbox["ymin"],
            "bbox_xmax": bbox["xmax"],
            "bbox_ymax": bbox["ymax"],
            "detected_at": datetime.now().isoformat()
        })

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def to_csv(self, file_path):
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)
        print(f"Saved {len(df)} records to {file_path}")
