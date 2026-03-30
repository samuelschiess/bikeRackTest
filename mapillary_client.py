import os
import requests
from io import BytesIO
from PIL import Image
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class MapillaryClient:
    def __init__(self, access_token: Optional[str] = None):
        """
        Initializes the Mapillary client with your developer access token.
        """
        self.access_token = access_token or os.environ.get("MAPILLARY_ACCESS_TOKEN")
        if not self.access_token or self.access_token == "your_token_here":
            raise ValueError("Mapillary Access Token is required. Please set it in the .env file.")
        
        self.base_url = "https://graph.mapillary.com"

    def get_images_in_bbox(self, bbox: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch a list of image metadata within a bounding box.
        
        :param bbox: A string of "min_lon,min_lat,max_lon,max_lat" (e.g., "-122.4,37.7,-122.3,37.8")
        :param limit: Max number of images to fetch
        :return: List of dictionaries containing image metadata
        """
        url = f"{self.base_url}/images"
        params = {
            "access_token": self.access_token,
            "fields": "id,geometry,computed_geometry,sequence,captured_at",
            "bbox": bbox,
            "limit": limit
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Mapillary API Error fetching bbox {bbox}: {e}")
            return []

    def download_image(self, image_id: str) -> Optional[Image.Image]:
        """
        Downloads a Mapillary image as a PIL Image object (in-memory) given its image_id.
        
        :param image_id: The mapillary graph ID of the image
        :return: PIL.Image object or None if failed
        """
        url = f"{self.base_url}/{image_id}"
        params = {
            "access_token": self.access_token,
            "fields": "thumb_2048_url"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            image_url = response.json().get("thumb_2048_url")
            if not image_url:
                return None

            # Download the actual image bytes
            img_response = requests.get(image_url)
            img_response.raise_for_status()

            return Image.open(BytesIO(img_response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            print(f"Mapillary API Error downloading image {image_id}: {e}")
            return None
        except Exception as e:
            print(f"Error reading image {image_id}: {e}")
            return None
