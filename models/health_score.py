from typing import Dict, Any, List
import numpy as np

class HealthScoreCalculator:
    """
    Road Condition Index (RCI) / Road Health Score Engine (0-100).
    Calculates unified ML-based road condition grade per segment.
    """
    def __init__(self):
        self.severity_deductions = {
            "LOW": 3.0,
            "MEDIUM": 7.5,
            "HIGH": 15.0,
            "CRITICAL": 25.0
        }

    def compute_score(
        self,
        road_id: str,
        detections: List[Dict[str, Any]],
        road_area_m2: float = 100.0,
        road_age_years: float = 3.0,
        previous_score: float = None
    ) -> Dict[str, Any]:
        """
        Compute total Health Index score (0 to 100) for a given road segment.
        """
        base_score = 100.0
        
        total_potholes = 0
        total_damaged_area = 0.0
        max_crack_density = 0.0

        for d in detections:
            damage_type = d.get("damage_type", "Pothole")
            severity = d.get("severity", "MEDIUM")
            metrics = d.get("metrics", {})
            
            if "pothole" in damage_type.lower():
                total_potholes += 1
                
            total_damaged_area += metrics.get("area_m2", 0.5)
            max_crack_density = max(max_crack_density, metrics.get("crack_density_pct", 0.0))
            
            # Apply severity-weighted deduction
            deduction = self.severity_deductions.get(severity, 5.0)
            base_score -= deduction

        # Deduct based on overall damage coverage percentage
        coverage_pct = (total_damaged_area / road_area_m2) * 100 if road_area_m2 > 0 else 0.0
        base_score -= (coverage_pct * 1.5)

        # Deduct based on road age factor
        base_score -= min(road_age_years * 1.2, 10.0)

        final_score = int(np.clip(round(base_score), 0, 100))

        # Categorize condition status
        if final_score >= 85:
            status = "GOOD"
            color = "GREEN"
        elif final_score >= 60:
            status = "FAIR"
            color = "YELLOW"
        elif final_score >= 35:
            status = "POOR"
            color = "ORANGE"
        else:
            status = "CRITICAL"
            color = "RED"

        return {
            "road_id": road_id,
            "road_health_score": final_score,
            "condition_status": status,
            "status_color": color,
            "breakdown": {
                "pothole_count": total_potholes,
                "total_damage_count": len(detections),
                "total_damaged_area_m2": round(total_damaged_area, 2),
                "max_crack_density_pct": round(max_crack_density, 2),
                "coverage_pct": round(coverage_pct, 2),
                "road_age_years": road_age_years,
                "previous_score": previous_score
            }
        }
