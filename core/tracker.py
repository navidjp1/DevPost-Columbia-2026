"""Football match tracker -- detection, tracking, team assignment, ball possession.

Closely follows the approach from:
    https://github.com/zakroum-hicham/football-analysis-CV

Pipeline:
    YOLO detection → NMS → ByteTrack → Team colour KMeans → Ball-to-player
    assignment → (optional) perspective transform via field keypoint model.

Works on CPU, MPS (Apple Silicon), and CUDA.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from sklearn.cluster import KMeans
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Class IDs for custom-trained model (same order as Roboflow dataset)
CLASS_BALL = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Represents a single tracked object across all frames."""

    object_id: int
    label: str  # "team_a", "team_b", "ball", "referee", "goalkeeper"
    # Per-frame data: frame_index -> data
    boxes: dict[int, np.ndarray] = field(default_factory=dict)  # [x1,y1,x2,y2]
    scores: dict[int, float] = field(default_factory=dict)

    def get_centroid(self, frame_index: int) -> tuple[float, float] | None:
        if frame_index not in self.boxes:
            return None
        box = self.boxes[frame_index]
        return (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))

    def get_foot_position(self, frame_index: int) -> tuple[float, float] | None:
        """Bottom-centre of the bounding box (more stable for ground plane)."""
        if frame_index not in self.boxes:
            return None
        box = self.boxes[frame_index]
        return (float((box[0] + box[2]) / 2), float(box[3]))


@dataclass
class TrackingResult:
    """Container for all tracking results from a video analysis."""

    objects: list[TrackedObject]
    total_frames: int
    fps: float
    width: int
    height: int
    ball_possession: dict[str, int] = field(default_factory=dict)

    def get_objects_by_label(self, label: str) -> list[TrackedObject]:
        return [obj for obj in self.objects if obj.label == label]

    def get_ball(self) -> TrackedObject | None:
        balls = self.get_objects_by_label("ball")
        return balls[0] if balls else None

    def get_object_by_id(self, object_id: int) -> TrackedObject | None:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_player_ids(self) -> list[int]:
        return [
            obj.object_id
            for obj in self.objects
            if obj.label in ("team_a", "team_b")
        ]


# -----------------------------------------------------------------------
# Team colour assignment (matches the reference implementation)
# -----------------------------------------------------------------------

class TeamAssigner:
    """Assign players to teams based on jersey colour using KMeans.

    Uses the top-half of each player bbox, clusters pixels into 2 groups
    (jersey vs background), picks the jersey cluster using corner pixels,
    then clusters all players into 2 teams.
    """

    def __init__(self):
        self.team_kmeans = None
        self.team_colors: dict[int, np.ndarray] = {}

    def get_player_color(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract the dominant jersey colour for one player."""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(3)

        top_half = crop[: crop.shape[0] // 2, :]
        if top_half.size == 0:
            top_half = crop

        hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 4:
            return np.zeros(3)

        km = KMeans(n_clusters=2, n_init=1, random_state=42)
        km.fit(pixels)

        labels_img = km.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corners = [
            labels_img[0, 0], labels_img[0, -1],
            labels_img[-1, 0], labels_img[-1, -1],
        ]
        bg_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - bg_cluster
        return km.cluster_centers_[player_cluster]

    def fit_teams(self, frame: np.ndarray, player_boxes: np.ndarray):
        """Cluster all visible players into 2 teams on the first frame."""
        colors = []
        for bbox in player_boxes:
            colors.append(self.get_player_color(frame, bbox))
        if len(colors) < 2:
            self.team_kmeans = None
            return

        self.team_kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto", random_state=42)
        self.team_kmeans.fit(colors)
        self.team_colors[0] = self.team_kmeans.cluster_centers_[0]
        self.team_colors[1] = self.team_kmeans.cluster_centers_[1]

    def get_team_id(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """Return 0 or 1 for a player."""
        if self.team_kmeans is None:
            return 0
        color = self.get_player_color(frame, bbox)
        return int(self.team_kmeans.predict(color.reshape(1, -1))[0])


# -----------------------------------------------------------------------
# Ball-to-player assignment (distance-based, from the reference)
# -----------------------------------------------------------------------

MAX_BALL_DISTANCE = 70  # pixels

def _assign_ball_to_player(
    player_boxes: np.ndarray,
    ball_center: tuple[float, float] | None,
) -> int:
    """Return the index of the closest player to the ball, or -1."""
    if ball_center is None or len(player_boxes) == 0:
        return -1

    bx, by = ball_center
    min_dist = MAX_BALL_DISTANCE
    best = -1

    for i, bbox in enumerate(player_boxes):
        x1, y1, x2, y2 = bbox
        # Distance from ball to both feet (bottom-left and bottom-right)
        dl = np.sqrt((x1 - bx) ** 2 + (y2 - by) ** 2)
        dr = np.sqrt((x2 - bx) ** 2 + (y2 - by) ** 2)
        d = min(dl, dr)
        if d < min_dist:
            min_dist = d
            best = i

    return best


# -----------------------------------------------------------------------
# Ball interpolation
# -----------------------------------------------------------------------

def _interpolate_ball(
    ball_obj: TrackedObject, total_frames: int, max_gap: int = 20
) -> None:
    """Fill short gaps in ball tracking via linear interpolation."""
    if not ball_obj.boxes:
        return

    rows = []
    for f in range(total_frames):
        if f in ball_obj.boxes:
            b = ball_obj.boxes[f]
            rows.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        else:
            rows.append([None, None, None, None])

    df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
    df = df.interpolate(method="linear", limit=max_gap, limit_area="inside")

    for f in range(total_frames):
        row = df.iloc[f]
        if row.isna().any():
            continue
        ball_obj.boxes[f] = np.array(
            [row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32
        )


# -----------------------------------------------------------------------
# Main tracker
# -----------------------------------------------------------------------

class SoccerTracker:
    """End-to-end football tracker.

    Supports two model sources:
      - A custom-trained YOLO model (best.pt from training/train.py)
      - The Roboflow inference model (fallback via API key)

    Usage::

        tracker = SoccerTracker(model_path="models/best.pt")
        result = tracker.analyze("clip.mp4")
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.3,
        device: str | None = None,
    ):
        self._model_path = model_path  # custom-trained model
        self._model_name = model_name  # generic YOLO fallback
        self._confidence = confidence
        self._device = device
        self._model: YOLO | None = None
        self._rf_model = None
        self._use_roboflow = False
        self._use_custom = False

    # ---- model loading ----

    def _load_model(self):
        if self._model is not None:
            return
        path = self._model_path or self._model_name
        logger.info(f"Loading YOLO model: {path}")
        self._model = YOLO(path)
        self._use_custom = self._model_path is not None
        logger.info("YOLO model loaded.")

    def _load_roboflow_model(self, api_key: str, model_id: str):
        if self._rf_model is not None:
            return
        os.environ["ROBOFLOW_API_KEY"] = api_key
        from inference import get_model
        self._rf_model = get_model(model_id=model_id)
        self._use_roboflow = True
        logger.info("Roboflow model loaded (local inference).")

    # ---- detection ----

    def _detect(self, frame: np.ndarray, conf: float):
        """Run detection and return sv.Detections."""
        if self._use_roboflow:
            results = self._rf_model.infer(frame, confidence=conf)
            if not results:
                return sv.Detections.empty()
            return sv.Detections.from_inference(results[0])
        else:
            results = self._model.predict(
                frame, conf=conf, device=self._device, verbose=False,
            )
            if not results or results[0].boxes is None:
                return sv.Detections.empty()
            return sv.Detections.from_ultralytics(results[0])

    # ---- analysis ----

    def analyze(
        self,
        video_path: str,
        track_ball: bool = True,
        fps: float | None = None,
        confidence: float | None = None,
        roboflow_api_key: str | None = None,
        roboflow_model_id: str = "football-players-detection-3zvbc/11",
        stride: int = 1,
        progress_callback: Any | None = None,
    ) -> TrackingResult:
        # Load model
        if roboflow_api_key and not self._model_path:
            if progress_callback:
                progress_callback("Loading Roboflow model...")
            self._load_roboflow_model(roboflow_api_key, roboflow_model_id)
        else:
            self._load_model()

        conf = confidence if confidence is not None else self._confidence

        # Video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        fps = fps or detected_fps

        # Class name → ID mapping (depends on model source)
        # For custom-trained models, class IDs are 0-3.
        # For Roboflow inference, we use class_name strings.

        # ByteTrack trackers
        team1_tracker = sv.ByteTrack(track_activation_threshold=conf, frame_rate=int(fps))
        team2_tracker = sv.ByteTrack(track_activation_threshold=conf, frame_rate=int(fps))
        ball_tracker = sv.ByteTrack(
            track_activation_threshold=max(conf - 0.1, 0.1), frame_rate=int(fps)
        ) if track_ball else None

        # Team assigner
        team_assigner = TeamAssigner()
        is_first_player_frame = True

        # Tracking state
        team_a_map: dict[int, TrackedObject] = {}
        team_b_map: dict[int, TrackedObject] = {}
        ball_map: dict[int, TrackedObject] = {}
        ref_map: dict[int, TrackedObject] = {}
        gk_map: dict[int, TrackedObject] = {}

        ball_possession: dict[str, int] = {"team_a": 0, "team_b": 0}
        last_possessing_team: str | None = None

        stride = max(1, stride)

        if progress_callback:
            progress_callback("Running detection and tracking...")

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if stride > 1 and frame_idx % stride != 0:
                frame_idx += 1
                continue

            if progress_callback and frame_idx % 30 == 0:
                pct = int(frame_idx / max(total_frames, 1) * 100)
                progress_callback(f"Processing frame {frame_idx}/{total_frames} ({pct}%)...")

            # ---- Detect ----
            detections = self._detect(frame, conf)

            if len(detections) == 0:
                frame_idx += 1
                continue

            # ---- Split by class ----
            class_names = detections.data.get("class_name", None)

            if class_names is not None and len(class_names) > 0:
                # Roboflow inference model (class names are strings)
                player_mask = np.array([
                    n.lower() in ("player", "goalkeeper") for n in class_names
                ], dtype=bool)
                ball_mask = np.array([
                    n.lower() == "ball" for n in class_names
                ], dtype=bool)
                ref_mask = np.array([
                    n.lower() == "referee" for n in class_names
                ], dtype=bool)
                gk_mask = np.array([
                    n.lower() == "goalkeeper" for n in class_names
                ], dtype=bool)
            else:
                # Custom-trained model (class IDs are integers)
                cids = detections.class_id
                player_mask = (cids == CLASS_PLAYER) | (cids == CLASS_GOALKEEPER)
                ball_mask = cids == CLASS_BALL
                ref_mask = cids == CLASS_REFEREE
                gk_mask = cids == CLASS_GOALKEEPER

            player_dets = detections[player_mask]
            ball_dets = detections[ball_mask]
            ref_dets = detections[ref_mask]

            # Apply NMS to players
            if len(player_dets) > 0:
                player_dets = player_dets.with_nms(threshold=0.5)

            # ---- Team assignment ----
            if len(player_dets) > 0:
                if is_first_player_frame and len(player_dets) >= 2:
                    team_assigner.fit_teams(frame, player_dets.xyxy)
                    is_first_player_frame = False

                if team_assigner.team_kmeans is not None:
                    team1_indices = []
                    team2_indices = []
                    for i in range(len(player_dets)):
                        tid = team_assigner.get_team_id(frame, player_dets.xyxy[i])
                        if tid == 0:
                            team1_indices.append(i)
                        else:
                            team2_indices.append(i)

                    t1_dets = player_dets[team1_indices] if team1_indices else sv.Detections.empty()
                    t2_dets = player_dets[team2_indices] if team2_indices else sv.Detections.empty()
                else:
                    t1_dets = player_dets
                    t2_dets = sv.Detections.empty()
            else:
                t1_dets = sv.Detections.empty()
                t2_dets = sv.Detections.empty()

            # ---- Track ----
            if len(t1_dets) > 0:
                t1_tracked = team1_tracker.update_with_detections(t1_dets)
                self._store_tracked(team_a_map, t1_tracked, frame_idx, "team_a")
            if len(t2_dets) > 0:
                t2_tracked = team2_tracker.update_with_detections(t2_dets)
                self._store_tracked(team_b_map, t2_tracked, frame_idx, "team_b")
            if track_ball and ball_tracker and len(ball_dets) > 0:
                b_tracked = ball_tracker.update_with_detections(ball_dets)
                self._store_tracked(ball_map, b_tracked, frame_idx, "ball")
            if len(ref_dets) > 0:
                # Store referee directly (no separate tracker needed for small count)
                for i in range(len(ref_dets)):
                    rid = i  # simple indexing per frame
                    if rid not in ref_map:
                        ref_map[rid] = TrackedObject(object_id=rid, label="referee")
                    ref_map[rid].boxes[frame_idx] = ref_dets.xyxy[i].copy()

            # ---- Ball-to-player assignment ----
            ball_center = None
            if track_ball and len(ball_dets) > 0:
                bx = ball_dets.xyxy[0]
                ball_center = (float((bx[0] + bx[2]) / 2), float((bx[1] + bx[3]) / 2))

            if ball_center is not None and len(player_dets) > 0:
                idx = _assign_ball_to_player(player_dets.xyxy, ball_center)
                if idx != -1:
                    # Determine which team this player belongs to
                    if team_assigner.team_kmeans is not None:
                        tid = team_assigner.get_team_id(frame, player_dets.xyxy[idx])
                        team_key = "team_a" if tid == 0 else "team_b"
                    else:
                        team_key = "team_a"
                    ball_possession[team_key] += 1
                    last_possessing_team = team_key
                elif last_possessing_team:
                    ball_possession[last_possessing_team] += 1
            elif last_possessing_team:
                ball_possession[last_possessing_team] += 1

            frame_idx += 1

        cap.release()

        # ---- Ball interpolation ----
        if track_ball and ball_map:
            if progress_callback:
                progress_callback("Interpolating ball positions...")
            best_tid = max(ball_map, key=lambda t: len(ball_map[t].boxes))
            best_ball = ball_map[best_tid]
            _interpolate_ball(best_ball, frame_idx)
            ball_map = {best_tid: best_ball}

        if progress_callback:
            progress_callback("Tracking complete!")

        # ---- Build final object list ----
        all_objects: list[TrackedObject] = []
        nid = 0
        for obj in team_a_map.values():
            obj.object_id = nid
            all_objects.append(obj)
            nid += 1
        for obj in team_b_map.values():
            obj.object_id = nid
            all_objects.append(obj)
            nid += 1
        for obj in ball_map.values():
            obj.object_id = nid
            all_objects.append(obj)
            nid += 1
        for obj in ref_map.values():
            obj.object_id = nid
            all_objects.append(obj)
            nid += 1

        return TrackingResult(
            objects=all_objects,
            total_frames=frame_idx,
            fps=fps,
            width=width,
            height=height,
            ball_possession=ball_possession,
        )

    @staticmethod
    def _store_tracked(
        tracked_map: dict[int, TrackedObject],
        detections: sv.Detections,
        frame_idx: int,
        label: str,
    ):
        if detections.tracker_id is None:
            return
        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            if tid not in tracked_map:
                tracked_map[tid] = TrackedObject(object_id=tid, label=label)
            obj = tracked_map[tid]
            obj.boxes[frame_idx] = detections.xyxy[i].copy()
            if detections.confidence is not None and i < len(detections.confidence):
                obj.scores[frame_idx] = float(detections.confidence[i])
