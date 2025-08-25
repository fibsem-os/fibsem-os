#!/usr/bin/env python3
"""
Client example for remote segmentation inference.
"""

import base64
import io
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional
import tifffile as tff

from fibsem.structures import FibsemImage


class RemoteSegmentationClient:
    """Client for remote segmentation inference"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check server health"""
        response = self.session.get(f"{self.server_url}/health")
        response.raise_for_status()
        return response.json()
    
    def encode_image(self, img: np.ndarray) -> str:
        """Encode numpy array to base64 string"""
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def decode_mask(self, encoded_mask: str) -> np.ndarray:
        """Decode base64 mask to numpy array"""
        mask_data = base64.b64decode(encoded_mask)
        mask_img = Image.open(io.BytesIO(mask_data))
        return np.array(mask_img)
    
    def segment(self, img: np.ndarray, return_rgb: bool = True) -> Optional[np.ndarray]:
        """Perform segmentation on image"""
        # Encode image
        encoded_image = self.encode_image(img)
        
        # Prepare request
        request_data = {
            "image": encoded_image,
            "return_rgb": return_rgb
        }
        
        # Send request
        response = self.session.post(
            f"{self.server_url}/segment",
            json=request_data
        )
        response.raise_for_status()
        
        result = response.json()
        
        if not result["success"]:
            raise Exception(f"Segmentation failed: {result.get('error', 'Unknown error')}")
        
        # Decode mask
        if result["mask"]:
            return self.decode_mask(result["mask"])
        
        return None
    
    def segment_fibsem_image(self, fibsem_img: FibsemImage, return_rgb: bool = True) -> Optional[np.ndarray]:
        """Segment a FibsemImage and return the mask"""
        return self.segment(fibsem_img.data, return_rgb=return_rgb)


def example_usage():
    """Example usage with test_image.tif"""
    
    # Initialize client
    client = RemoteSegmentationClient("http://localhost:8000")
    
    # Check server health
    try:
        health = client.health_check()
        from pprint import pprint
        pprint(health)
        print(f"Server status: {health['status']}")
        print(f"GPU available: {health['gpu']['available']}")
        print(f"Model loaded: {health['model']['loaded']}")
        print(f"Uptime: {health['stats']['uptime_seconds']:.1f}s")
        print(f"Memory usage: {health['system']['memory_percent']:.1f}%")
        if health['gpu']['available']:
            print(f"GPU devices: {health['gpu']['device_count']}")
    except requests.exceptions.RequestException as e:
        print(f"Server not available: {e}")
        return
    
    # Load test image
    import fibsem
    test_image_path = os.path.join(os.path.dirname(fibsem.__path__[0]), "fibsem", "segmentation", "test_image.tif")
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        return
    
    # Load the test image
    test_image = tff.imread(test_image_path)
    print(f"Loaded test image: {test_image.shape}, dtype: {test_image.dtype}")
    
    try:
        # Perform segmentation
        print("Performing segmentation...")
        mask = client.segment(test_image, return_rgb=False)

        if mask is not None:
            print(f"Segmentation successful!")
            print(f"Input shape: {test_image.shape}")
            print(f"Output shape: {mask.shape}")
            print(f"Output dtype: {mask.dtype}")
            
            # Plot results
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            axes[0].imshow(test_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Segmentation mask
            if len(mask.shape) == 3:  # RGB mask
                axes[1].imshow(mask)
            else:  # Grayscale mask
                axes[1].imshow(mask, cmap='viridis')
            axes[1].set_title('Segmentation Mask')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("No mask returned")
    
    except Exception as e:
        print(f"Segmentation failed: {e}")


if __name__ == "__main__":
    example_usage()