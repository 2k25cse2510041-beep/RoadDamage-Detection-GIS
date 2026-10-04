import numpy as np
import cv2
from typing import Dict, Any, List
from models.detection import DamageDetector
from models.segmentation import RoadAndDamageSegmentor
from models.classification import DamageClassifier
from models.measurement import DamageMeasurer
from models.severity import SeverityPredictor
from models.health_score import HealthScoreCalculator
from models.tracking import VideoTracker
from models.spatial_gis import SpatialClusterer
from models.infrastructure import InfrastructureDetector
from models.predictive import FutureDamagePredictor
from models.explainability import ModelExplainability
from models.active_learning import ActiveLearningPipeline
from models.spatio_temporal import SpatioTemporalGraphModel
from models.report_generator import ReportGenerator

class RoadVisionPipeline:
    """
    Master ML Pipeline orchestrating all 28+ ML capabilities for RoadVision AI.
    """
    def __init__(self):
        self.detector = DamageDetector()
        self.segmentor = RoadAndDamageSegmentor()
        self.classifier = DamageClassifier()
        self.measurer = DamageMeasurer()
        self.severity_predictor = SeverityPredictor()
        self.health_calculator = HealthScoreCalculator()
        self.tracker = VideoTracker()
        self.spatial_clusterer = SpatialClusterer()
        self.infra_detector = InfrastructureDetector()
        self.predictive_engine = FutureDamagePredictor()
        self.explainability_engine = ModelExplainability()
        self.active_learning = ActiveLearningPipeline()
        self.digital_twin_engine = SpatioTemporalGraphModel()
        self.report_generator = ReportGenerator()

    def process_image(
        self,
        image: np.ndarray,
        road_id: str = "R102",
        latitude: float = 28.6139,
        longitude: float = 77.2090,
        historical_trend: str = "Increasing"
    ) -> Dict[str, Any]:
        """
        Run end-to-end ML inference on a single inspection image frame.
        """
        if image is None:
            raise ValueError("Input image is None")

        h, w = image.shape[:2]

        # 1. Road Surface Segmentation
        road_mask, road_area_m2 = self.segmentor.segment_road(image)

        # 2. Damage Object Detection
        raw_detections = self.detector.detect(image)

        # 3. Process each detection through Classification, Segmentation, Measurement & Severity
        processed_detections = []
        total_damaged_area = 0.0

        for d in raw_detections:
            bbox = d["bbox"]
            initial_type = d["damage_type"]
            
            x1, y1, x2, y2 = bbox
            crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            cls_info = self.classifier.classify_crop(crop, initial_label=initial_type)
            final_type = cls_info["predicted_class"]

            dmg_mask, dmg_area_m2 = self.segmentor.segment_damage(image, bbox)
            total_damaged_area += dmg_area_m2

            metrics = self.measurer.measure(dmg_mask, final_type, bbox)

            severity_info = self.severity_predictor.predict(
                damage_type=final_type,
                area_m2=metrics["area_m2"],
                crack_density_pct=metrics.get("crack_density_pct", 0.0),
                damage_count=len(raw_detections),
                historical_trend=historical_trend
            )

            explanation = self.explainability_engine.explain_severity(
                severity_info,
                {"area_m2": metrics["area_m2"], "crack_density_pct": metrics.get("crack_density_pct", 0.0), "damage_count": len(raw_detections)}
            )

            detection_entry = {
                "damage_type": final_type,
                "confidence": d["confidence"],
                "bbox": bbox,
                "severity": severity_info["severity"],
                "severity_confidence": severity_info["confidence"],
                "metrics": metrics,
                "explanation": explanation,
                "latitude": latitude,
                "longitude": longitude,
                "road_id": road_id
            }
            processed_detections.append(detection_entry)

        # 4. Damage Coverage Calculation
        coverage_metrics = self.segmentor.compute_damage_coverage(road_area_m2, total_damaged_area)

        # 5. Multi-task Infrastructure, Traffic & Waterlogging Analysis
        scene_analysis = self.infra_detector.analyze_scene(image)

        # 6. Road Health Index Score
        health_info = self.health_calculator.compute_score(
            road_id=road_id,
            detections=processed_detections,
            road_area_m2=road_area_m2
        )

        # 7. Weather-Aware Predictive Risk & Deterioration Model
        predictive_info = self.predictive_engine.predict_risk(
            road_id=road_id,
            current_health_score=health_info["road_health_score"],
            damage_count=len(processed_detections),
            total_damaged_area_m2=total_damaged_area
        )

        # 8. Active Learning Uncertainty Sampling
        uncertain_queue = self.active_learning.sample_uncertainty(processed_detections)

        # 9. Road Digital Twin Profile
        digital_twin = self.digital_twin_engine.get_digital_twin_profile(
            road_id=road_id,
            health_score=health_info["road_health_score"],
            detections=processed_detections,
            gis_coords=(latitude, longitude)
        )

        # 10. Automated Inspection Report Synthesis
        report = self.report_generator.generate_report(
            road_id=road_id,
            road_health=health_info,
            detections=processed_detections,
            predictive_info=predictive_info,
            gis_info=(latitude, longitude)
        )

        return {
            "road_id": road_id,
            "road_health": health_info,
            "coverage": coverage_metrics,
            "detections": processed_detections,
            "scene_analysis": scene_analysis,
            "predictive_forecasting": predictive_info,
            "digital_twin": digital_twin,
            "active_learning_uncertain_samples": len(uncertain_queue),
            "report": report
        }

    def process_video_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Process a single frame from video stream with tracking & duplicate removal.
        """
        detections = self.detector.detect(frame)
        tracked_detections = self.tracker.update_frame(frame_number, detections)
        return {
            "frame_number": frame_number,
            "raw_detection_count": len(detections),
            "unique_tracked_objects": tracked_detections
        }
