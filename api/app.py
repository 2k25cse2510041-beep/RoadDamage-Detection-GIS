import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from pipeline.inference_pipeline import RoadVisionPipeline
from models.spatial_gis import SpatialClusterer
from models.active_learning import ActiveLearningPipeline
from config.settings import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="REST API interface for RoadVision AI ML engine (Detection, Segmentation, GIS, Predictive Risk & Digital Twin)."
)

# Enable CORS for Frontend (React, Vue, Angular, Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RoadVisionPipeline()
clusterer = SpatialClusterer()
active_learning = ActiveLearningPipeline()

class GISPoint(BaseModel):
    road_id: str = "R101"
    latitude: float
    longitude: float
    damage_type: str = "Pothole"
    severity: str = "HIGH"
    confidence: float = 0.95

class FeedbackRequest(BaseModel):
    sample_id: str
    corrected_label: str
    is_valid_detection: bool = True

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "supported_damage_classes": settings.DAMAGE_CLASSES
    }

@app.post("/api/v1/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    road_id: str = Form("R102"),
    latitude: float = Form(28.6139),
    longitude: float = Form(77.2090),
    historical_trend: str = Form("Increasing")
):
    """
    Primary endpoint for single image inspection.
    Executes end-to-end ML pipeline: Detection -> Seg -> Measure -> Severity -> Health -> Prediction -> Report.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")

        result = pipeline.process_image(
            image=img,
            road_id=road_id,
            latitude=latitude,
            longitude=longitude,
            historical_trend=historical_trend
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict-video-frame")
async def predict_video_frame(
    file: UploadFile = File(...),
    frame_number: int = Form(1)
):
    """
    Endpoint for sequential video frames with ByteTrack tracking & duplicate detection.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame format")

        tracking_result = pipeline.process_video_frame(frame, frame_number)
        return tracking_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/spatial-clustering")
def run_spatial_clustering(points: List[GISPoint]):
    """
    Clusters GPS points into High Damage Clusters (DBSCAN) and returns GeoJSON features.
    """
    dict_points = [p.dict() for p in points]
    clustering_res = clusterer.cluster_points(dict_points)
    return clustering_res

@app.get("/api/v1/digital-twin/{road_id}")
def get_road_digital_twin(road_id: str):
    """
    Fetches the ML-generated Digital Twin profile for a given road ID.
    """
    profile = pipeline.digital_twin_engine.get_digital_twin_profile(road_id=road_id)
    return profile

@app.post("/api/v1/active-learning/feedback")
def submit_human_feedback(request: FeedbackRequest):
    """
    Ingest human feedback / corrections into the active learning loop.
    """
    res = active_learning.record_human_feedback(
        sample_id=request.sample_id,
        corrected_label=request.corrected_label,
        is_valid=request.is_valid_detection
    )
    return res

@app.get("/api/v1/model-metrics")
def get_model_performance_metrics():
    """
    Returns current model accuracy, mAP, precision, recall, and versioning info.
    """
    return active_learning.get_model_metrics()
