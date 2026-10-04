import os
import cv2
import numpy as np
import json
from typing import Tuple, List, Dict, Any

def generate_test_data(output_dir: str = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generates a realistic test road image with synthetic potholes/cracks
    and a sample dataset of GIS road inspection points for testing.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "sample_road.jpg")

    # 1. Create a 640x480 RGB road texture image
    h, w = 480, 640
    # Asphalt dark gray background
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    
    # Add random asphalt noise texture
    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw lane markings (dashed white line in center)
    for y in range(0, h, 60):
        cv2.rectangle(img, (w // 2 - 4, y), (w // 2 + 4, y + 35), (220, 220, 220), -1)

    # Synthetic Pothole (dark oval region with rough border)
    cv2.ellipse(img, (220, 310), (55, 35), 15, 0, 360, (25, 25, 25), -1)
    cv2.ellipse(img, (220, 310), (58, 38), 15, 0, 360, (10, 10, 10), 3)

    # Synthetic Crack (jagged line network)
    pts = np.array([[400, 180], [420, 220], [410, 260], [450, 310], [480, 350]], np.int32)
    cv2.polylines(img, [pts], False, (15, 15, 15), 4)

    cv2.imwrite(img_path, img)

    # 2. Generate sample GIS dataset (nearby GPS points for spatial clustering test)
    base_lat, base_lng = 28.6139, 77.2090
    gis_dataset = [
        {"road_id": "R101", "latitude": base_lat + 0.0001, "longitude": base_lng + 0.0002, "damage_type": "Pothole", "severity": "HIGH", "confidence": 0.94},
        {"road_id": "R101", "latitude": base_lat + 0.0002, "longitude": base_lng + 0.0003, "damage_type": "Crack", "severity": "MEDIUM", "confidence": 0.91},
        {"road_id": "R101", "latitude": base_lat + 0.00015, "longitude": base_lng + 0.00025, "damage_type": "Pothole", "severity": "CRITICAL", "confidence": 0.96},
        {"road_id": "R102", "latitude": base_lat + 0.008, "longitude": base_lng + 0.009, "damage_type": "Edge damage", "severity": "LOW", "confidence": 0.88},
        {"road_id": "R102", "latitude": base_lat + 0.0082, "longitude": base_lng + 0.0091, "damage_type": "Rutting", "severity": "MEDIUM", "confidence": 0.90}
    ]

    gis_file = os.path.join(output_dir, "sample_gis_data.json")
    with open(gis_file, "w") as f:
        json.dump(gis_dataset, f, indent=2)

    return img_path, gis_dataset

if __name__ == "__main__":
    path, gis = generate_test_data()
    print(f"Generated sample image: {path}")
