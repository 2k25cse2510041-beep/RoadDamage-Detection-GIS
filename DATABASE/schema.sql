-- =========================================================
-- AI Road Damage Intelligence Platform — Database Schema
-- PostgreSQL + PostGIS
-- =========================================================

-- Make sure PostGIS is enabled (safe to re-run)
CREATE EXTENSION IF NOT EXISTS postgis;

-- =========================================================
-- 1. USERS
-- =========================================================
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(150) NOT NULL,
    email           VARCHAR(150) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(50) NOT NULL DEFAULT 'inspector'
                        CHECK (role IN ('admin', 'inspector', 'municipal_staff')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================================================
-- 2. ROADS
-- =========================================================
CREATE TABLE roads (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    segment_code    VARCHAR(100),
    geom            GEOMETRY(LineString, 4326),  -- road segment path, optional
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_roads_geom ON roads USING GIST (geom);

-- =========================================================
-- 3. DAMAGE TYPES (lookup table)
-- =========================================================
CREATE TABLE damage_types (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(50) NOT NULL UNIQUE
                        CHECK (name IN ('Pothole', 'Alligator Crack', 'Linear Crack', 'Other'))
);

INSERT INTO damage_types (name) VALUES
    ('Pothole'), ('Alligator Crack'), ('Linear Crack'), ('Other');

-- =========================================================
-- 4. MEDIA (images and videos uploaded for inspection)
-- =========================================================
CREATE TABLE media (
    id              SERIAL PRIMARY KEY,
    road_id         INTEGER REFERENCES roads(id) ON DELETE SET NULL,
    uploaded_by     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    media_type      VARCHAR(10) NOT NULL CHECK (media_type IN ('image', 'video')),
    file_path       VARCHAR(500) NOT NULL,   -- S3/local storage reference
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_media_road_id ON media(road_id);
CREATE INDEX idx_media_uploaded_at ON media(uploaded_at);

-- =========================================================
-- 5. DETECTIONS (core ML output table)
-- =========================================================
CREATE TABLE detections (
    id                  SERIAL PRIMARY KEY,
    media_id            INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    road_id             INTEGER REFERENCES roads(id) ON DELETE SET NULL,
    damage_type_id      INTEGER NOT NULL REFERENCES damage_types(id),

    confidence          FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    severity            VARCHAR(10) NOT NULL CHECK (severity IN ('Low', 'Medium', 'High')),

    -- bounding box (pixel coordinates in the source frame)
    bbox_x              FLOAT,
    bbox_y              FLOAT,
    bbox_width          FLOAT,
    bbox_height         FLOAT,

    damage_area         FLOAT,                 -- optional estimated area/size
    frame_no            INTEGER,               -- required for video detections
    detected_at         TIMESTAMPTZ,            -- frame timestamp / capture time

    geom                GEOMETRY(Point, 4326),  -- spatial location (lon/lat, WGS84)

    model_version       VARCHAR(50) NOT NULL,
    status              VARCHAR(20) NOT NULL DEFAULT 'Reported'
                            CHECK (status IN ('Reported', 'In-Review', 'Fixed')),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_detections_media_id ON detections(media_id);
CREATE INDEX idx_detections_road_id ON detections(road_id);
CREATE INDEX idx_detections_damage_type ON detections(damage_type_id);
CREATE INDEX idx_detections_severity ON detections(severity);
CREATE INDEX idx_detections_created_at ON detections(created_at);
CREATE INDEX idx_detections_geom ON detections USING GIST (geom);

-- =========================================================
-- 6. MAINTENANCE RECORDS
-- =========================================================
CREATE TABLE maintenance_records (
    id                  SERIAL PRIMARY KEY,
    detection_id        INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    priority             VARCHAR(10) NOT NULL DEFAULT 'Medium'
                            CHECK (priority IN ('Low', 'Medium', 'High')),
    maintenance_status   VARCHAR(20) NOT NULL DEFAULT 'Pending'
                            CHECK (maintenance_status IN ('Pending', 'Assigned', 'In-Progress', 'Completed')),
    reported_date        DATE NOT NULL DEFAULT CURRENT_DATE,
    assigned_team        VARCHAR(150),
    repair_date           DATE,
    completion_status     BOOLEAN NOT NULL DEFAULT FALSE,
    repair_notes           TEXT
);

CREATE INDEX idx_maintenance_detection_id ON maintenance_records(detection_id);
CREATE INDEX idx_maintenance_status ON maintenance_records(maintenance_status);

-- =========================================================
-- Done. Quick sanity check:
-- =========================================================
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
