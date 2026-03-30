import os
from mapillary_client import MapillaryClient

def collect_dataset(count=1000):
    mly_client = MapillaryClient()
    os.makedirs("dataset_raw", exist_ok=True)
    
    print("Beginning Mapillary Harvest for Custom Dataset...")
    
    # We define a larger bounding box across Downtown Salt Lake City
    bbox = "-111.910,40.750,-111.870,40.780"
    
    print(f"Fetching images for bbox: {bbox}")
    
    images_meta = mly_client.get_images_in_bbox(bbox, limit=count)
    print(f"Found {len(images_meta)} images in bounding box. Downloading...")
    
    downloaded = 0
    for i, meta in enumerate(images_meta):
        image_id = meta.get("id")
        
        # Don't re-download if we already have it!
        if os.path.exists(f"dataset_raw/{image_id}.jpg"):
            print(f"[{i+1}/{len(images_meta)}] Image {image_id} already exists! Skipping...")
            downloaded += 1
            continue
            
        print(f"[{i+1}/{len(images_meta)}] Downloading image {image_id}...")
        
        try:
            img = mly_client.download_image(image_id)
            if img:
                # Save just the raw image without drawing any boxes
                img.save(f"dataset_raw/{image_id}.jpg")
                downloaded += 1
        except Exception as e:
            print(f"Failed to download {image_id}: {e}")
            
    print(f"\nHarvest complete! Successfully downloaded {downloaded} raw images to 'dataset_raw'.")
    print("Next step: Upload this folder to Roboflow (roboflow.com) or CVAT to manually draw bounding boxes!")

if __name__ == "__main__":
    # We will aim for 200 images to start building the dataset
    collect_dataset(count=200)
