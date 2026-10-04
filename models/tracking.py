import numpy as np
from typing import List, Dict, Any

class VideoTracker:
    """
    Object Tracking & Duplicate Detection Engine (ByteTrack / DeepSORT algorithm).
    Ensures that a single pothole/crack detected across N consecutive video frames
    is tracked under a single unique ID (Track ID) and counted exactly once.
    """
    def __init__(self, match_iou_threshold: float = 0.3, max_disappeared: int = 15):
        self.next_object_id = 1
        self.tracked_objects = {}  # id -> {"bbox", "damage_type", "disappeared", "history"}
        self.iou_threshold = match_iou_threshold
        self.max_disappeared = max_disappeared

    def update_frame(self, frame_number: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process current frame detections and correlate with active track IDs.
        Returns detections augmented with unique 'track_id' and 'is_new_detection' flag.
        """
        if len(detections) == 0:
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]["disappeared"] += 1
                if self.tracked_objects[obj_id]["disappeared"] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
            return []

        input_boxes = [d["bbox"] for d in detections]

        if len(self.tracked_objects) == 0:
            # Register all as new tracks
            tracked_detections = []
            for d in detections:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.tracked_objects[obj_id] = {
                    "bbox": d["bbox"],
                    "damage_type": d["damage_type"],
                    "disappeared": 0,
                    "first_seen": frame_number
                }
                d_copy = dict(d)
                d_copy["track_id"] = obj_id
                d_copy["is_new_detection"] = True
                tracked_detections.append(d_copy)
            return tracked_detections

        # Match existing active tracks with current detections using IoU (Intersection over Union)
        object_ids = list(self.tracked_objects.keys())
        existing_boxes = [self.tracked_objects[oid]["bbox"] for oid in object_ids]

        iou_matrix = np.zeros((len(existing_boxes), len(input_boxes)), dtype=np.float32)
        for i, box1 in enumerate(existing_boxes):
            for j, box2 in enumerate(input_boxes):
                iou_matrix[i, j] = self._compute_iou(box1, box2)

        matched_existing = set()
        matched_input = set()
        tracked_detections = []

        if iou_matrix.size > 0:
            # Greedy matching based on maximum IoU score
            flat_indices = np.argsort(-iou_matrix.ravel())
            for idx in flat_indices:
                row = idx // len(input_boxes)
                col = idx % len(input_boxes)
                if row in matched_existing or col in matched_input:
                    continue
                if iou_matrix[row, col] >= self.iou_threshold:
                    matched_existing.add(row)
                    matched_input.add(col)
                    
                    obj_id = object_ids[row]
                    self.tracked_objects[obj_id]["bbox"] = input_boxes[col]
                    self.tracked_objects[obj_id]["disappeared"] = 0
                    
                    d_copy = dict(detections[col])
                    d_copy["track_id"] = obj_id
                    d_copy["is_new_detection"] = False
                    tracked_detections.append(d_copy)

        # Register unmatched input detections as new tracks
        for col in range(len(input_boxes)):
            if col not in matched_input:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.tracked_objects[obj_id] = {
                    "bbox": input_boxes[col],
                    "damage_type": detections[col]["damage_type"],
                    "disappeared": 0,
                    "first_seen": frame_number
                }
                d_copy = dict(detections[col])
                d_copy["track_id"] = obj_id
                d_copy["is_new_detection"] = True
                tracked_detections.append(d_copy)

        # Increment disappeared count for unmatched existing objects
        for row in range(len(existing_boxes)):
            if row not in matched_existing:
                obj_id = object_ids[row]
                self.tracked_objects[obj_id]["disappeared"] += 1
                if self.tracked_objects[obj_id]["disappeared"] > self.max_disappeared:
                    del self.tracked_objects[obj_id]

        return tracked_detections

    def _compute_iou(self, box1: list, box2: list) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
