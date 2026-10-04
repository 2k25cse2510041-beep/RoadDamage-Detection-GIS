import time
from typing import Dict, Any, List

class ReportGenerator:
    """
    Multimodal AI Inspection Reporting Engine.
    Generates:
    1. Structured JSON API payload.
    2. Executive Natural Language Text Report.
    3. Markdown Formatted Inspection Document (Ready for export / PDF conversion).
    """
    def __init__(self):
        pass

    def generate_report(
        self,
        road_id: str,
        road_health: Dict[str, Any],
        detections: List[Dict[str, Any]],
        predictive_info: Dict[str, Any] = None,
        gis_info: tuple = (28.6139, 77.2090)
    ) -> Dict[str, Any]:
        """
        Synthesize full inspection report across all ML modules.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC")
        score = road_health.get("road_health_score", 75)
        status = road_health.get("condition_status", "FAIR")
        
        # Natural Language Summary Synthesis
        pothole_count = sum(1 for d in detections if "pothole" in d.get("damage_type", "").lower())
        crack_count = sum(1 for d in detections if "crack" in d.get("damage_type", "").lower())
        total_area = sum(d.get("metrics", {}).get("area_m2", 0.5) for d in detections)
        
        exec_summary = (
            f"Automated AI visual inspection for Road Segment '{road_id}' completed on {timestamp}. "
            f"The segment received a Road Condition Index score of {score}/100 ({status}). "
            f"A total of {len(detections)} damage instances were identified ({pothole_count} Potholes, {crack_count} Cracks) "
            f"affecting an estimated surface area of {total_area:.2f} m². "
        )

        if predictive_info:
            m_rec = predictive_info.get("maintenance_recommendation", {})
            action = m_rec.get("action", "Routine Monitoring")
            priority = m_rec.get("priority_category", "MEDIUM")
            risk_30 = predictive_info.get("risk_forecast", {}).get("day_30_risk_pct", 50.0)
            
            exec_summary += (
                f"Historical deterioration modeling indicates a 30-day failure risk of {risk_30}%. "
                f"Recommended maintenance intervention: '{action}' with priority category '{priority}'."
            )

        # Markdown Document Generation
        markdown_doc = f"""# ROAD INSPECTION REPORT

**Road Segment ID:** {road_id}  
**Inspection Date:** {timestamp}  
**GPS Coordinates:** Lat {gis_info[0]}, Lng {gis_info[1]}  

---

### 📊 Executive Summary
{exec_summary}

---

### 🛣️ Road Condition Metrics
* **Road Health Score:** `{score}/100` ({status})
* **Total Damages Identified:** `{len(detections)}`
* **Total Affected Surface Area:** `{total_area:.2f} m²`
* **Primary Damage Types:** Potholes ({pothole_count}), Cracks ({crack_count})

---

### 🚨 Detected Damage Inventory
"""
        for idx, d in enumerate(detections, 1):
            dtype = d.get("damage_type", "Pothole")
            sev = d.get("severity", "MEDIUM")
            conf = d.get("confidence", 0.90) * 100
            area = d.get("metrics", {}).get("area_m2", 0.5)
            markdown_doc += f"{idx}. **{dtype}** | Severity: `{sev}` | Area: `{area} m²` | Confidence: `{conf:.1f}%`\n"

        if predictive_info:
            m_rec = predictive_info.get("maintenance_recommendation", {})
            markdown_doc += f"""
---

### 🔮 Predictive AI & Maintenance Recommendation
* **30-Day Failure Risk:** `{predictive_info.get('risk_forecast', {}).get('day_30_risk_pct')}%`
* **90-Day Failure Risk:** `{predictive_info.get('risk_forecast', {}).get('day_90_risk_pct')}%`
* **Recommended Action:** `{m_rec.get('action')}`
* **Priority Level:** `{m_rec.get('priority_category')}`
"""

        markdown_doc += "\n---\n*Report generated automatically by RoadVision AI ML Core Engine*"

        return {
            "road_id": road_id,
            "timestamp": timestamp,
            "executive_summary": exec_summary,
            "markdown_report": markdown_doc,
            "structured_payload": {
                "road_health": road_health,
                "detections": detections,
                "predictive": predictive_info
            }
        }
