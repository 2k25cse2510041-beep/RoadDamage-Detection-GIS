import os

class Settings:
    """Global configuration settings for RoadVision AI ML engine."""
    PROJECT_NAME: str = "RoadVision AI - ML Intelligence Core"
    VERSION: str = "2.0.0"

    # Supported Damage Categories
    DAMAGE_CLASSES = [
        "Pothole",
        "Crack",
        "Alligator crack",
        "Longitudinal crack",
        "Transverse crack",
        "Rutting",
        "Road patch",
        "Depression",
        "Edge damage"
    ]

    # Infrastructure & Obstruction Categories
    INFRASTRUCTURE_CLASSES = [
        "Manhole",
        "Speed breaker",
        "Traffic sign",
        "Traffic signal",
        "Divider",
        "Street light",
        "Zebra crossing",
        "Road barricade"
    ]

    OBSTRUCTION_CLASSES = [
        "Construction debris",
        "Open manhole",
        "Road blockage",
        "Construction material",
        "Temporary barriers"
    ]

    # Detection & Confidence Thresholds
    DETECTION_CONF_THRESHOLD: float = 0.40
    TRACKING_MATCH_THRESHOLD: float = 0.50
    ACTIVE_LEARNING_UNCERTAINTY_MIN: float = 0.45
    ACTIVE_LEARNING_UNCERTAINTY_MAX: float = 0.65

    # Pixel to Meter Calibration Default (e.g. 1 pixel ~ 0.005 meters depending on pitch/altitude)
    PIXELS_PER_METER_DEFAULT: float = 200.0  # 200 pixels = 1 meter

    # Severity Thresholds (Area in m², Crack Density %)
    SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    # Spatial Clustering (DBSCAN params - distance in meters/GPS delta ~ 0.001 deg ~ 100m)
    DBSCAN_EPS_KM: float = 0.1  # 100 meters
    DBSCAN_MIN_SAMPLES: int = 3

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

settings = Settings()
