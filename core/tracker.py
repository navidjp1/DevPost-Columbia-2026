"""YOLOv8 + ByteTrack video tracking for soccer match analysis.

Uses a fine-tuned Roboflow football detection model (run locally via the
``inference`` package) or a generic YOLOv8 fallback for per-frame object
detection, ByteTrack for consistent ID tracking across frames, and K-Means
jersey color clustering for automatic team classification.
Works on CPU, MPS (Apple Silicon), and CUDA.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs used by YOLOv8
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32

# Roboflow football-players-detection model class names (lowercase)
RF_PLAYER_CLASSES = {"player", "goalkeeper"}
RF_BALL_CLASSES = {"ball"}
RF_REFEREE_CLASSES = {"referee"}


@dataclass
class TrackedObject:
    """Represents a single tracked object across all frames."""

    object_id: int
    label: str  # "team_a", "team_b", or "ball"
    # Per-frame data: frame_index -> data
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    boxes: dict[int, np.ndarray] = field(default_factory=dict)  # [x1, y1, x2, y2]
    scores: dict[int, float] = field(default_factory=dict)

    def get_centroid(self, frame_index: int) -> tuple[float, float] | None:
        """Get the centroid (cx, cy) of this object's bounding box at a frame.

        Returns None if the object was not detected in that frame.
        """
        if frame_index not in self.boxes:
            return None
        box = self.boxes[frame_index]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return (float(cx), float(cy))


@dataclass
class TrackingResult:
    """Container for all tracking results from a video analysis session."""

    objects: list[TrackedObject]
    total_frames: int
    fps: float
    width: int
    height: int

    def get_objects_by_label(self, label: str) -> list[TrackedObject]:
        """Get all tracked objects with a given label."""
        return [obj for obj in self.objects if obj.label == label]

    def get_ball(self) -> TrackedObject | None:
        """Get the ball object, if tracked."""
        balls = self.get_objects_by_label("ball")
        return balls[0] if balls else None

    def get_object_by_id(self, object_id: int) -> TrackedObject | None:
        """Get a tracked object by its ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_player_ids(self) -> list[int]:
        """Get all player object IDs (excludes ball)."""
        return [
            obj.object_id
            for obj in self.objects
            if obj.label in ("team_a", "team_b")
        ]


class SoccerTracker:
    """High-level tracker using YOLOv8/Roboflow + ByteTrack + jersey color clustering.

    Detects players and ball, tracks them with persistent IDs, and
    automatically classifies players into two teams based on jersey color.

    Usage:
        tracker = SoccerTracker()
        result = tracker.analyze(video_path="clip.mp4", track_ball=True)
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str | None = None,
        confidence: float = 0.3,
    ):
        """Initialize the tracker.

        Args:
            model_name: YOLOv8 model name/path (used when Roboflow is not active).
            device: Device string ("cpu", "mps", "cuda"). Auto-detected if None.
            confidence: Minimum detection confidence threshold (0-1).
        """
        self._model_name = model_name
        self._model: YOLO | None = None
        self._device = device
        self._confidence = confidence

        # Roboflow model (loaded lazily if API key provided)
        self._rf_model = None
        self._use_roboflow = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazily load the YOLO model."""
        if self._model is not None:
            return
        logger.info(f"Loading YOLOv8 model: {self._model_name}")
        self._model = YOLO(self._model_name)
        logger.info("YOLOv8 model loaded successfully.")

    def _load_roboflow_model(self, api_key: str, model_id: str = "football-players-detection-3zvbc/11"):
        """Load the Roboflow football-players-detection model locally.

        Downloads the model weights on first run (requires API key) and
        caches them locally.  All subsequent inference runs entirely on
        device -- no network calls per frame.

        The default model detects: ball, goalkeeper, player, referee.

        Args:
            api_key: Roboflow API key (needed for initial download).
            model_id: Roboflow model ID in ``project/version`` format.
        """
        if self._rf_model is not None:
            return

        logger.info(f"Loading Roboflow model locally: {model_id} ...")
        # inference reads the key from the env var
        os.environ["ROBOFLOW_API_KEY"] = api_key

        from inference import get_model

        self._rf_model = get_model(model_id=model_id)
        self._use_roboflow = True
        logger.info("Roboflow model loaded successfully (local inference).")

    # ------------------------------------------------------------------
    # Jersey color team classification
    # ------------------------------------------------------------------

    @staticmethod
    def _get_dominant_jersey_color(
        frame: np.ndarray, box: np.ndarray
    ) -> np.ndarray:
        """Extract the dominant jersey color from a player bounding box.

        Crops the upper 60% of the player (jersey region), converts to
        HSV for better color discrimination, then uses K-Means to find
        the dominant non-green color (to avoid pitch contamination).

        Args:
            frame: Full BGR frame.
            box: [x1, y1, x2, y2] bounding box.

        Returns:
            Dominant color as a 3-element HSV numpy array.
        """
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return np.array([0.0, 0.0, 0.0])

        # Focus on upper body (jersey, not shorts/legs)
        h = player_crop.shape[0]
        jersey_crop = player_crop[: int(h * 0.6), :]
        if jersey_crop.size == 0:
            jersey_crop = player_crop

        # Convert to HSV for better color clustering
        hsv_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        pixels = hsv_crop.reshape(-1, 3).astype(np.float32)

        if len(pixels) < 10:
            return np.array([0.0, 0.0, 0.0])

        # K-Means to find dominant colors (3 clusters)
        n_clusters = min(3, len(pixels))
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        kmeans.fit(pixels)

        # Pick the most frequent cluster that is NOT green (pitch)
        counts = np.bincount(kmeans.labels_, minlength=n_clusters)
        centers = kmeans.cluster_centers_

        # Sort clusters by frequency (most common first)
        order = np.argsort(-counts)

        for idx in order:
            center = centers[idx]
            hue = center[0]  # HSV hue: 0-180 in OpenCV
            sat = center[1]

            # Skip green-ish colors (pitch grass): hue ~35-85, sat > 40
            is_green = 35 < hue < 85 and sat > 40
            if not is_green:
                return center

        # Fallback: return most common cluster
        return centers[order[0]]

    @staticmethod
    def _classify_teams(
        jersey_colors: list[np.ndarray],
        n_teams: int = 2,
    ) -> list[str]:
        """Cluster jersey colors into teams using K-Means.

        Args:
            jersey_colors: List of dominant HSV colors (one per player).
            n_teams: Number of teams to cluster into.

        Returns:
            List of labels: "team_a" or "team_b" for each player.
        """
        if len(jersey_colors) < 2:
            return ["team_a"] * len(jersey_colors)

        color_array = np.array(jersey_colors, dtype=np.float32)

        # Cluster into n_teams groups
        kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
        team_labels = kmeans.fit_predict(color_array)

        return ["team_a" if t == 0 else "team_b" for t in team_labels]

    # ------------------------------------------------------------------
    # Roboflow detection helpers
    # ------------------------------------------------------------------

    def _detect_roboflow(
        self, frame: np.ndarray, conf: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run detection using the local Roboflow model.

        The model runs entirely on-device (no network calls).  Accepts
        a numpy BGR array directly -- no temp-file I/O.

        Args:
            frame: BGR numpy array.
            conf: Confidence threshold (0-1).

        Returns:
            Tuple of (player_boxes, player_confs, player_classes,
                      ball_boxes, ball_confs, ball_classes).
        """
        empty_box = np.empty((0, 4), dtype=np.float32)
        empty_conf = np.empty(0, dtype=np.float32)
        empty_cls = np.empty(0, dtype=int)

        # inference.get_model().infer() accepts numpy arrays directly
        results = self._rf_model.infer(frame, confidence=conf)
        if not results:
            return empty_box, empty_conf, empty_cls, empty_box.copy(), empty_conf.copy(), empty_cls.copy()

        # Convert to supervision Detections for uniform handling
        detections = sv.Detections.from_inference(results[0])

        if len(detections) == 0:
            return empty_box, empty_conf, empty_cls, empty_box.copy(), empty_conf.copy(), empty_cls.copy()

        # Class names are stored in detections.data["class_name"]
        class_names = detections.data.get("class_name", np.array([]))

        player_mask = np.array(
            [name.lower() in RF_PLAYER_CLASSES for name in class_names],
            dtype=bool,
        )
        ball_mask = np.array(
            [name.lower() in RF_BALL_CLASSES for name in class_names],
            dtype=bool,
        )

        if player_mask.any():
            p_boxes = detections.xyxy[player_mask].astype(np.float32)
            p_confs = detections.confidence[player_mask].astype(np.float32)
            p_classes = np.zeros(int(player_mask.sum()), dtype=int)
        else:
            p_boxes, p_confs, p_classes = empty_box, empty_conf, empty_cls

        if ball_mask.any():
            b_boxes = detections.xyxy[ball_mask].astype(np.float32)
            b_confs = detections.confidence[ball_mask].astype(np.float32)
            b_classes = np.ones(int(ball_mask.sum()), dtype=int)
        else:
            b_boxes = empty_box.copy()
            b_confs = empty_conf.copy()
            b_classes = empty_cls.copy()

        return p_boxes, p_confs, p_classes, b_boxes, b_confs, b_classes

    def _detect_yolo(
        self, frame: np.ndarray, conf: float, classes_to_detect: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run detection using YOLOv8.

        Returns:
            Tuple of (player_boxes, player_confs, player_classes,
                      ball_boxes, ball_confs, ball_classes).
        """
        results = self._model.predict(
            frame,
            conf=conf,
            classes=classes_to_detect,
            device=self._device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            empty = np.empty((0, 4), dtype=np.float32)
            empty_c = np.empty(0, dtype=np.float32)
            empty_cls = np.empty(0, dtype=int)
            return empty, empty_c, empty_cls, empty.copy(), empty_c.copy(), empty_cls.copy()

        result = results[0]
        all_boxes = result.boxes.xyxy.cpu().numpy()
        all_confs = result.boxes.conf.cpu().numpy()
        all_classes = result.boxes.cls.cpu().numpy().astype(int)

        player_mask = all_classes == PERSON_CLASS_ID
        ball_mask = all_classes == SPORTS_BALL_CLASS_ID

        p_boxes = all_boxes[player_mask] if player_mask.any() else np.empty((0, 4), dtype=np.float32)
        p_confs = all_confs[player_mask] if player_mask.any() else np.empty(0, dtype=np.float32)
        p_classes = all_classes[player_mask] if player_mask.any() else np.empty(0, dtype=int)

        b_boxes = all_boxes[ball_mask] if ball_mask.any() else np.empty((0, 4), dtype=np.float32)
        b_confs = all_confs[ball_mask] if ball_mask.any() else np.empty(0, dtype=np.float32)
        b_classes = all_classes[ball_mask] if ball_mask.any() else np.empty(0, dtype=int)

        return p_boxes, p_confs, p_classes, b_boxes, b_confs, b_classes

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        video_path: str,
        track_ball: bool = True,
        fps: float | None = None,
        confidence: float | None = None,
        roboflow_api_key: str | None = None,
        roboflow_model_id: str = "football-players-detection-3zvbc/11",
        progress_callback: Any | None = None,
    ) -> TrackingResult:
        """Run full tracking analysis on a soccer video clip.

        Args:
            video_path: Path to the video file (MP4, AVI, etc.).
            track_ball: Whether to track the ball.
            fps: Video FPS (auto-detected from video if None).
            confidence: Override detection confidence for this run.
            roboflow_api_key: If provided, use Roboflow football model
                              (run locally) instead of generic YOLOv8.
            roboflow_model_id: Roboflow model ID (project/version).
            progress_callback: Optional callable(status_str) for progress updates.

        Returns:
            TrackingResult with all tracked objects and per-frame data.
        """
        # Load the appropriate model
        if roboflow_api_key:
            if progress_callback:
                progress_callback("Loading Roboflow football model (local)...")
            self._load_roboflow_model(roboflow_api_key, model_id=roboflow_model_id)
        else:
            self._load_model()

        conf = confidence if confidence is not None else self._confidence

        # ---- Read video metadata ----
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        fps = fps or detected_fps

        if progress_callback:
            progress_callback("Initializing trackers...")

        # ---- Set up ByteTrack trackers ----
        player_tracker = sv.ByteTrack(
            track_activation_threshold=conf,
            frame_rate=int(fps),
        )
        ball_tracker = sv.ByteTrack(
            track_activation_threshold=max(conf - 0.1, 0.1),
            frame_rate=int(fps),
        ) if track_ball else None

        # COCO classes for YOLO fallback
        classes_to_detect = [PERSON_CLASS_ID]
        if track_ball:
            classes_to_detect.append(SPORTS_BALL_CLASS_ID)

        # ---- Tracking state ----
        tracked_map: dict[int, TrackedObject] = {}
        ball_tracked_map: dict[int, TrackedObject] = {}

        # Collect jersey colors per tracker_id for team classification
        # tracker_id -> list of dominant HSV colors seen across frames
        jersey_color_samples: dict[int, list[np.ndarray]] = {}

        if progress_callback:
            progress_callback("Running detection and tracking...")

        # ---- Process each frame ----
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if progress_callback and frame_idx % 30 == 0:
                pct = int(frame_idx / max(total_frames, 1) * 100)
                progress_callback(
                    f"Processing frame {frame_idx}/{total_frames} ({pct}%)..."
                )

            # ---- Detect ----
            if self._use_roboflow:
                p_boxes, p_confs, p_cls, b_boxes, b_confs, b_cls = (
                    self._detect_roboflow(frame, conf)
                )
            else:
                p_boxes, p_confs, p_cls, b_boxes, b_confs, b_cls = (
                    self._detect_yolo(frame, conf, classes_to_detect)
                )

            # ---- Track players ----
            if len(p_boxes) > 0:
                player_detections = sv.Detections(
                    xyxy=p_boxes,
                    confidence=p_confs,
                    class_id=p_cls,
                )
                tracked_players = player_tracker.update_with_detections(
                    player_detections
                )

                # Update tracking state + sample jersey colors
                if tracked_players.tracker_id is not None:
                    for i, tracker_id in enumerate(tracked_players.tracker_id):
                        tid = int(tracker_id)
                        box = tracked_players.xyxy[i]

                        if tid not in tracked_map:
                            tracked_map[tid] = TrackedObject(
                                object_id=tid,
                                label="team_a",  # placeholder, assigned later
                            )
                        obj = tracked_map[tid]
                        obj.boxes[frame_idx] = box.copy()
                        if (
                            tracked_players.confidence is not None
                            and i < len(tracked_players.confidence)
                        ):
                            obj.scores[frame_idx] = float(
                                tracked_players.confidence[i]
                            )

                        # Sample jersey color (every 10 frames to save compute)
                        if frame_idx % 10 == 0:
                            color = self._get_dominant_jersey_color(frame, box)
                            if tid not in jersey_color_samples:
                                jersey_color_samples[tid] = []
                            jersey_color_samples[tid].append(color)

            # ---- Track ball ----
            if track_ball and ball_tracker is not None and len(b_boxes) > 0:
                ball_detections = sv.Detections(
                    xyxy=b_boxes,
                    confidence=b_confs,
                    class_id=b_cls,
                )
                tracked_balls = ball_tracker.update_with_detections(
                    ball_detections
                )
                self._update_tracked_objects(
                    ball_tracked_map, tracked_balls, frame_idx, label="ball"
                )

            frame_idx += 1

        cap.release()

        # ---- Classify teams by jersey color ----
        if progress_callback:
            progress_callback("Classifying teams by jersey color...")

        # Compute average jersey color per player
        avg_colors: dict[int, np.ndarray] = {}
        for tid, samples in jersey_color_samples.items():
            if samples:
                avg_colors[tid] = np.mean(samples, axis=0)

        # Get ordered list of tracker IDs that have color samples
        tids_with_color = [tid for tid in tracked_map if tid in avg_colors]
        if tids_with_color:
            color_list = [avg_colors[tid] for tid in tids_with_color]
            team_labels = self._classify_teams(color_list, n_teams=2)

            for tid, team_label in zip(tids_with_color, team_labels):
                tracked_map[tid].label = team_label

        # Players without enough color samples default to team_a
        for tid, obj in tracked_map.items():
            if tid not in avg_colors:
                obj.label = "team_a"

        if progress_callback:
            progress_callback("Tracking complete!")

        # ---- Build final object list with unique IDs ----
        all_objects: list[TrackedObject] = []
        next_id = 0

        for obj in tracked_map.values():
            obj.object_id = next_id
            all_objects.append(obj)
            next_id += 1

        for obj in ball_tracked_map.values():
            obj.object_id = next_id
            all_objects.append(obj)
            next_id += 1

        return TrackingResult(
            objects=all_objects,
            total_frames=frame_idx,
            fps=fps,
            width=width,
            height=height,
        )

    @staticmethod
    def _update_tracked_objects(
        tracked_map: dict[int, TrackedObject],
        detections: sv.Detections,
        frame_idx: int,
        label: str,
    ):
        """Update the tracked objects map with new detections for a frame.

        Args:
            tracked_map: Dict mapping tracker_id -> TrackedObject.
            detections: Supervision Detections with tracker_id set.
            frame_idx: Current frame index.
            label: Label for new objects.
        """
        if detections.tracker_id is None:
            return

        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)

            if tid not in tracked_map:
                tracked_map[tid] = TrackedObject(
                    object_id=tid,
                    label=label,
                )

            obj = tracked_map[tid]
            obj.boxes[frame_idx] = detections.xyxy[i].copy()

            if detections.confidence is not None and i < len(detections.confidence):
                obj.scores[frame_idx] = float(detections.confidence[i])
