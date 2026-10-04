import os
import cv2
import unittest
import numpy as np

from data.synthetic_generator import generate_test_data
from pipeline.inference_pipeline import RoadVisionPipeline
from models.spatial_gis import SpatialClusterer

class TestRoadVisionPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        img_path, gis_data = generate_test_data()
        image = cv2.imread(img_path)
        
        pipeline = RoadVisionPipeline()
        result = pipeline.process_image(image, road_id="R101")
        
        self.assertIn("road_health", result)
        self.assertIn("detections", result)
        self.assertIn("predictive_forecasting", result)
        self.assertIn("report", result)
        self.assertGreaterEqual(result["road_health"]["road_health_score"], 0)
        self.assertLessEqual(result["road_health"]["road_health_score"], 100)

    def test_spatial_clustering(self):
        clusterer = SpatialClusterer()
        sample_points = [
            {"road_id": "R101", "latitude": 28.6139, "longitude": 77.2090, "damage_type": "Pothole", "severity": "HIGH"},
            {"road_id": "R101", "latitude": 28.6140, "longitude": 77.2091, "damage_type": "Crack", "severity": "MEDIUM"},
            {"road_id": "R101", "latitude": 28.6141, "longitude": 77.2092, "damage_type": "Pothole", "severity": "CRITICAL"}
        ]
        res = clusterer.cluster_points(sample_points)
        self.assertEqual(res["total_damages"], 3)
        self.assertIn("geojson", res)

if __name__ == "__main__":
    unittest.main()
