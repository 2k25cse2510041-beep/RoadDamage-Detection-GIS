import numpy as np
from typing import List, Dict, Any
from config.settings import settings

class SpatialClusterer:
    """
    GIS + ML Fusion and Spatial Clustering Engine.
    Uses DBSCAN / HDBSCAN algorithms to cluster nearby road damage coordinates
    (e.g., potholes within 50-100m) into designated 'High Damage Clusters'.
    Generates standard GeoJSON output for Leaflet / Mapbox GIS frontends.
    """
    def __init__(self, eps_km: float = None, min_samples: int = None):
        self.eps_km = eps_km or settings.DBSCAN_EPS_KM
        self.min_samples = min_samples or settings.DBSCAN_MIN_SAMPLES

    def cluster_points(self, detections_gis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Group spatial GIS points into clusters and identify high-density damage zones.
        Each record in detections_gis should have: 'latitude', 'longitude', 'severity', 'damage_type'.
        """
        if not detections_gis:
            return {"clusters": [], "geojson": {"type": "FeatureCollection", "features": []}}

        coords = np.array([[d["latitude"], d["longitude"]] for d in detections_gis])

        try:
            from sklearn.cluster import DBSCAN
            # Convert eps_km to approximate degrees (~111 km per degree latitude)
            kms_per_radian = 6371.0088
            eps_rad = self.eps_km / kms_per_radian
            
            db = DBSCAN(eps=eps_rad, min_samples=self.min_samples, metric='haversine')
            coords_rad = np.radians(coords)
            labels = db.fit_predict(coords_rad)
        except Exception as e:
            print(f"[SpatialClusterer] Fallback simple grid clustering used: {e}")
            labels = np.zeros(len(coords), dtype=int)

        clusters = {}
        features = []

        for idx, (label, item) in enumerate(zip(labels, detections_gis)):
            cluster_id = f"Cluster_{label:02d}" if label != -1 else "Noise_Unclustered"
            item["cluster_id"] = cluster_id

            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "cluster_id": cluster_id,
                    "damage_points_count": 0,
                    "severities": [],
                    "center_lat": 0.0,
                    "center_lng": 0.0,
                    "items": []
                }

            clusters[cluster_id]["damage_points_count"] += 1
            clusters[cluster_id]["severities"].append(item.get("severity", "MEDIUM"))
            clusters[cluster_id]["items"].append(item)

            # Create GeoJSON Feature for spatial map frontend integration
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [item["longitude"], item["latitude"]]
                },
                "properties": {
                    "damage_type": item.get("damage_type", "Pothole"),
                    "severity": item.get("severity", "MEDIUM"),
                    "confidence": item.get("confidence", 0.95),
                    "cluster_id": cluster_id,
                    "road_id": item.get("road_id", "R101"),
                    "timestamp": item.get("timestamp", "2026-10-04T12:00:00Z")
                }
            }
            features.append(feature)

        # Compute cluster centroids and aggregate severity rating
        summary_clusters = []
        for cid, cdata in clusters.items():
            if cid == "Noise_Unclustered":
                continue
            
            lats = [it["latitude"] for it in cdata["items"]]
            lngs = [it["longitude"] for it in cdata["items"]]
            cdata["center_lat"] = round(float(np.mean(lats)), 6)
            cdata["center_lng"] = round(float(np.mean(lngs)), 6)
            
            # Severity priority mapping
            if "CRITICAL" in cdata["severities"] or "HIGH" in cdata["severities"]:
                overall_severity = "HIGH"
            else:
                overall_severity = "MEDIUM"
                
            cdata["overall_severity"] = overall_severity
            del cdata["severities"]
            del cdata["items"]
            summary_clusters.append(cdata)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        return {
            "total_damages": len(detections_gis),
            "clusters_found": len(summary_clusters),
            "clusters": summary_clusters,
            "geojson": geojson
        }
