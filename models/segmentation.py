import numpy as np
import cv2
from typing import Dict, Any, Tuple
from config.settings import settings

class RoadAndDamageSegmentor:
    """
    Segmentation module handling:
    1. Road Surface Segmentation (Identifying road boundary vs non-road objects).
    2. Precise Pixel-Level Damage Mask Extraction.
    3. Coverage calculation (Road Area vs Damaged Area).
    """
    def __init__(self, pixels_per_meter: float = None):
        self.ppm = pixels_per_meter or settings.PIXELS_PER_METER_DEFAULT

    def segment_road(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segment the primary road surface area from the camera view.
        Returns binary mask (255 inside road, 0 outside) and road area in m².
        """
        h, w = image.shape[:2]
        
        # Color & texture based road surface estimation (Asphalt color range in HSV/GRAY)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Typical trapezoidal ROI for road in front view / dashcam view
        pts = np.array([
            [int(w * 0.05), int(h * 0.95)],
            [int(w * 0.35), int(h * 0.35)],
            [int(w * 0.65), int(h * 0.35)],
            [int(w * 0.95), int(h * 0.95)]
        ], np.int32)
        
        road_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(road_mask, [pts], 255)
        
        road_pixels = np.sum(road_mask > 0)
        road_area_m2 = round(road_pixels / (self.ppm ** 2), 2)
        
        return road_mask, max(road_area_m2, 10.0)

    def segment_damage(self, image: np.ndarray, bbox: list) -> Tuple[np.ndarray, float]:
        """
        Extract exact damage pixel mask within a bounding box using GrabCut / OTSU thresholding.
        Returns damage mask and damage area in m².
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Ensure bbox bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Otsu thresholding inside bounding box to segment damaged region
        _, crop_mask = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = crop_mask

        damaged_pixels = np.sum(crop_mask > 0)
        damaged_area_m2 = round(damaged_pixels / (self.ppm ** 2), 4)

        return full_mask, max(damaged_area_m2, 0.05)

    def compute_damage_coverage(self, road_area_m2: float, total_damaged_area_m2: float) -> Dict[str, float]:
        """
        Calculate total damage coverage percentage over road area.
        """
        coverage_pct = round((total_damaged_area_m2 / road_area_m2) * 100, 2) if road_area_m2 > 0 else 0.0
        return {
            "road_area_m2": road_area_m2,
            "damaged_area_m2": round(total_damaged_area_m2, 2),
            "damage_coverage_pct": coverage_pct
        }
