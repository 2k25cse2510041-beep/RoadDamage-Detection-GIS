import os
import cv2
import json
import numpy as np

from data.synthetic_generator import generate_test_data
from pipeline.inference_pipeline import RoadVisionPipeline
from models.spatial_gis import SpatialClusterer
from config.settings import settings

def run_demo():
    print("===============================================================")
    print(" [ML] ROADVISION AI - FULL END-TO-END ML DEMONSTRATION")
    print("===============================================================")

    # Step 1: Generate synthetic test image and sample GIS dataset
    data_dir = os.path.join(settings.BASE_DIR, "data")
    img_path, gis_data = generate_test_data(data_dir)
    print(f"\n[1/5] Sample road image & GIS dataset initialized at: {img_path}")

    # Step 2: Initialize ML Pipeline
    pipeline = RoadVisionPipeline()
    print("[2/5] ML Core Engine Loaded (Detection, Seg, Severity, Health, Predictive, XAI).")

    # Step 3: Run full inference pipeline on test image
    image = cv2.imread(img_path)
    result = pipeline.process_image(
        image=image,
        road_id="R102",
        latitude=28.6139,
        longitude=77.2090,
        historical_trend="Increasing"
    )
    print("[3/5] Inference Pipeline Execution Complete.")

    # Step 4: Draw bounding boxes and severity annotations on output image
    output_dir = os.path.join(settings.BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    annotated_img = image.copy()
    for d in result["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        dtype = d["damage_type"]
        sev = d["severity"]
        conf = d["confidence"]
        
        # Color coding by severity (BGR format)
        color_map = {"LOW": (0, 255, 0), "MEDIUM": (0, 255, 255), "HIGH": (0, 165, 255), "CRITICAL": (0, 0, 255)}
        color = color_map.get(sev, (0, 0, 255))
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        label = f"{dtype} [{sev}] {conf*100:.1f}%"
        cv2.putText(annotated_img, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    annotated_out_path = os.path.join(output_dir, "annotated_road_output.jpg")
    cv2.imwrite(annotated_out_path, annotated_img)
    print(f"[4/5] Annotated visualization saved to: {annotated_out_path}")

    # Step 5: Perform Spatial GIS Clustering (DBSCAN)
    spatial_clusterer = SpatialClusterer()
    clustering_res = spatial_clusterer.cluster_points(gis_data)
    print("[5/5] Spatial GIS DBSCAN Clustering Completed.")

    # Print Summary Results
    print("\n---------------------------------------------------------------")
    print(" [RESULTS] ROAD HEALTH & DAMAGE METRICS SUMMARY")
    print("---------------------------------------------------------------")
    health = result["road_health"]
    print(f"Road ID               : {health['road_id']}")
    print(f"Road Health Score     : {health['road_health_score']} / 100 ({health['condition_status']})")
    print(f"Total Damages Count   : {len(result['detections'])}")
    print(f"Damage Coverage       : {result['coverage']['damage_coverage_pct']}% of road surface")
    print(f"Waterlogging State    : {result['scene_analysis']['waterlogging']['condition']}")
    
    print("\n---------------------------------------------------------------")
    print(" [PREDICTIVE] RISK & MAINTENANCE RECOMMENDATION")
    print("---------------------------------------------------------------")
    pred = result["predictive_forecasting"]
    m_rec = pred["maintenance_recommendation"]
    print(f"30-Day Failure Risk   : {pred['risk_forecast']['day_30_risk_pct']}%")
    print(f"90-Day Failure Risk   : {pred['risk_forecast']['day_90_risk_pct']}%")
    print(f"Progression Trend     : {pred['progression_trend']}")
    print(f"Recommended Action    : {m_rec['action']}")
    print(f"Priority Category     : {m_rec['priority_category']}")

    print("\n---------------------------------------------------------------")
    print(" [GIS] SPATIAL CLUSTERING RESULTS")
    print("---------------------------------------------------------------")
    print(f"Total GIS Points      : {clustering_res['total_damages']}")
    print(f"Clusters Identified   : {clustering_res['clusters_found']}")
    for c in clustering_res["clusters"]:
        print(f" -> {c['cluster_id']}: {c['damage_points_count']} points | Center: ({c['center_lat']}, {c['center_lng']}) | Severity: {c['overall_severity']}")

    print("\n---------------------------------------------------------------")
    print(" [REPORT] GENERATED AI INSPECTION REPORT (SNIPPET)")
    print("---------------------------------------------------------------")
    print(result["report"]["executive_summary"])
    
    # Save full JSON result payload
    json_out_path = os.path.join(output_dir, "full_inspection_results.json")
    json_payload = {
        "road_id": result["road_id"],
        "road_health": result["road_health"],
        "coverage": result["coverage"],
        "detections": result["detections"],
        "predictive_forecasting": result["predictive_forecasting"],
        "digital_twin": result["digital_twin"],
        "clustering_results": clustering_res,
        "executive_summary": result["report"]["executive_summary"]
    }
    with open(json_out_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\n[SUCCESS] Full inspection JSON payload saved to: {json_out_path}")
    print("===============================================================\n")

if __name__ == "__main__":
    run_demo()
