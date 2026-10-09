-- =========================================================
-- Sample seed data for road_damage_db
-- Run this AFTER schema.sql
-- =========================================================

-- 1. A sample user (inspector)
INSERT INTO users (name, email, password_hash, role)
VALUES ('Pranjal Pandey', 'pranjalwork749@gmail.com', 'hashed_password_placeholder', 'inspector');

-- 2. A sample road
INSERT INTO roads (name, segment_code)
VALUES ('MG Road', 'MGR-001');

-- 3. A sample media upload (pretend image)
INSERT INTO media (road_id, uploaded_by, media_type, file_path)
VALUES (1, 1, 'image', '/uploads/sample_road_image_1.jpg');

-- 4. Sample detections with realistic Lucknow-area coordinates
--    damage_type_id: 1=Pothole, 2=Alligator Crack, 3=Linear Crack, 4=Other
INSERT INTO detections
    (media_id, road_id, damage_type_id, confidence, severity,
     bbox_x, bbox_y, bbox_width, bbox_height,
     damage_area, frame_no, detected_at, geom, model_version, status)
VALUES
    (1, 1, 1, 0.91, 'High',
     120, 340, 80, 60,
     4800, NULL, now(),
     ST_SetSRID(ST_MakePoint(80.9462, 26.8467), 4326),  -- lon, lat (Lucknow)
     'yolov8n-v1', 'Reported'),

    (1, 1, 3, 0.78, 'Medium',
     200, 150, 150, 20,
     3000, NULL, now(),
     ST_SetSRID(ST_MakePoint(80.9500, 26.8500), 4326),
     'yolov8n-v1', 'Reported'),

    (1, 1, 2, 0.85, 'High',
     50, 400, 200, 100,
     20000, NULL, now(),
     ST_SetSRID(ST_MakePoint(80.9400, 26.8400), 4326),
     'yolov8n-v1', 'In-Review'),

    (1, 1, 1, 0.65, 'Low',
     300, 100, 40, 30,
     1200, NULL, now(),
     ST_SetSRID(ST_MakePoint(80.9480, 26.8480), 4326),
     'yolov8n-v1', 'Reported');

-- 5. A maintenance record for one of the high-severity detections
INSERT INTO maintenance_records
    (detection_id, priority, maintenance_status, assigned_team)
VALUES
    (1, 'High', 'Assigned', 'Municipal Roads Team A');

-- =========================================================
-- Sanity check: see your seeded detections
-- =========================================================
-- SELECT id, damage_type_id, severity, ST_AsText(geom) AS location FROM detections;
