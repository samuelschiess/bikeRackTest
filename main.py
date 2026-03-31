import os
import argparse
import random
from PIL import ImageDraw
from mapillary_client import MapillaryClient
from detector import YoloWorldDetector
from storage import BikeRackDataStore

def run_pipeline(bbox: str, n_images: int = 8, seed: int | None = None) -> None:
    print("Initialize Mapillary Client...")
    mly_client = MapillaryClient()
    
    print("Initialize Detectors...")
    # YOLO-World scores tend to be lower out of the box, setting threshold to 0.05
    yolo_detector = YoloWorldDetector(threshold=0.05)
    
    store = BikeRackDataStore()

    # Create directories for output images
    os.makedirs("output_images/yolo/detected", exist_ok=True)
    os.makedirs("output_images/yolo/not_detected", exist_ok=True)

    print(f"Fetching images for bbox: {bbox}")
    images_meta = mly_client.get_images_in_bbox(bbox, limit=1000)
    
    if seed is not None:
        print(f"Using random seed: {seed}")
        random.seed(seed)
        
    if n_images and n_images < len(images_meta):
        images_meta = random.sample(images_meta, n_images)
        print(f"Randomly selected {n_images} images. Processing...")
    else:
        print(f"Found {len(images_meta)} images. Processing...")

    for i, meta in enumerate(images_meta):
        image_id = meta.get("id")
        
        geom = meta.get("computed_geometry") or meta.get("geometry")
        if not geom:
            continue
            
        lon, lat = geom["coordinates"]
        captured_at = meta.get("captured_at")
        
        print(f"[{i+1}/{len(images_meta)}] Downloading image {image_id}...")
        try:
            img = mly_client.download_image(image_id)
            if not img:
                continue
        except Exception as e:
            print(f"Error downloading image: {e}")
            continue
            
        # ---------------------------------------------
        # RUN YOLO-WORLD
        # ---------------------------------------------
        try:
            img_yolo = img.copy()
            yolo_preds = yolo_detector.detect(img_yolo)
            
            if yolo_preds:
                print(f"  -> YOLO-World found {len(yolo_preds)} bike racks.")
                draw = ImageDraw.Draw(img_yolo)
                for i_box, d in enumerate(yolo_preds):
                    box = d["box"]
                    score = d["score"]
                    print(f"     Rack {i_box+1}: Confidence {score:.2f}")
                    
                    draw.rectangle([(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])], outline="red", width=5)
                    
                    text = f"Rack {i_box+1} ({score:.2f})"
                    text_x = box["xmin"]
                    text_y = max(0, box["ymin"] - 15)
                    draw.rectangle([(text_x, text_y), (text_x + 90, text_y + 15)], fill="red")
                    draw.text((text_x + 2, text_y + 2), text, fill="white")
                    
                    filename = f"output_images/yolo/detected/{image_id}.jpg"
                    store.add_record(image_id, "YOLO-World", lat, lon, captured_at, score, box, os.path.abspath(filename))
                
                img_yolo.save(f"output_images/yolo/detected/{image_id}.jpg")
            else:
                print("  -> YOLO-World found 0 bike racks.")
                img_yolo.save(f"output_images/yolo/not_detected/{image_id}.jpg")
        except Exception as e:
            print(f"Error in YOLO-World: {e}")

    print("\nPipeline Complete!")
    df = store.to_dataframe()
    print("Results Summary:")
    if not df.empty:
        print(df.groupby("model_name").size().reset_index(name="Total Detections"))
        store.to_csv("bike_racks_output.csv")
    else:
        print("No bike racks detected by either model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mapillary Bike Rack Detector")
    parser.add_argument(
        "--bbox", 
        type=str, 
        default="-111.903,40.764,-111.901,40.765",
        help="Bounding box 'min_lon,min_lat,max_lon,max_lat' (default is test area near Chile Tepin in SLC)"
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=8,
        help="Number of random images to process (default: 8). Set to 0 to process all."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for image sampling reproducibility (default: None)"
    )
    args = parser.parse_args()
    
    run_pipeline(args.bbox, n_images=args.n_images, seed=args.seed)
