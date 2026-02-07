"""YOLOv8 + ByteTrack video tracking for soccer match analysis.

Uses YOLOv8 for per-frame object detection (players and ball)
and ByteTrack (via supervision) for consistent ID tracking across frames.
Works on CPU, MPS (Apple Silicon), and CUDA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs used by YOLOv8
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32


@dataclass
class TrackedObject:
    """Represents a single tracked object across all frames."""

    object_id: int
    label: str  # "player" or "ball"
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
            if obj.label == "player"
        ]


class SoccerTracker:
    """High-level tracker using YOLOv8 + ByteTrack to track players and ball.

    Usage:
        tracker = SoccerTracker()
        result = tracker.analyze(
            video_path="clip.mp4",
            track_ball=True,
        )
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str | None = None,
        confidence: float = 0.3,
    ):
        """Initialize the tracker.

        Args:
            model_name: YOLOv8 model name/path. Options:
                        "yolov8n.pt" (nano, fastest),
                        "yolov8s.pt" (small),
                        "yolov8m.pt" (medium, most accurate).
            device: Device string ("cpu", "mps", "cuda"). Auto-detected if None.
            confidence: Minimum detection confidence threshold (0-1).
        """
        self._model_name = model_name
        self._model: YOLO | None = None
        self._device = device  # None means YOLO auto-detects
        self._confidence = confidence

    def _load_model(self):
        """Lazily load the YOLO model."""
        if self._model is not None:
            return

        logger.info(f"Loading YOLOv8 model: {self._model_name}")
        self._model = YOLO(self._model_name)
        logger.info("YOLOv8 model loaded successfully.")

    def analyze(
        self,
        video_path: str,
        track_ball: bool = True,
        fps: float | None = None,
        confidence: float | None = None,
        progress_callback: Any | None = None,
    ) -> TrackingResult:
        """Run full tracking analysis on a soccer video clip.

        Args:
            video_path: Path to the video file (MP4, AVI, etc.).
            track_ball: Whether to track the ball.
            fps: Video FPS (auto-detected from video if None).
            confidence: Override detection confidence for this run.
            progress_callback: Optional callable(status_str) for progress updates.

        Returns:
            TrackingResult with all tracked objects and per-frame data.
        """
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

        # ---- Which COCO classes to detect ----
        classes_to_detect = [PERSON_CLASS_ID]
        if track_ball:
            classes_to_detect.append(SPORTS_BALL_CLASS_ID)

        # ---- Process each frame ----
        # Dict: tracker_id -> TrackedObject
        tracked_map: dict[int, TrackedObject] = {}
        # Separate namespace for ball IDs to avoid collisions
        ball_tracked_map: dict[int, TrackedObject] = {}

        if progress_callback:
            progress_callback("Running detection and tracking...")

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

            # Run YOLO detection
            results = self._model.predict(
                frame,
                conf=conf,
                classes=classes_to_detect,
                device=self._device,
                verbose=False,
            )

            if not results or results[0].boxes is None:
                frame_idx += 1
                continue

            result = results[0]
            all_boxes = result.boxes.xyxy.cpu().numpy()
            all_confs = result.boxes.conf.cpu().numpy()
            all_classes = result.boxes.cls.cpu().numpy().astype(int)

            # ---- Split detections: players vs ball ----
            player_mask = all_classes == PERSON_CLASS_ID
            ball_mask = all_classes == SPORTS_BALL_CLASS_ID

            # -- Track players --
            if player_mask.any():
                player_detections = sv.Detections(
                    xyxy=all_boxes[player_mask],
                    confidence=all_confs[player_mask],
                    class_id=all_classes[player_mask],
                )
                tracked_players = player_tracker.update_with_detections(
                    player_detections
                )
                self._update_tracked_objects(
                    tracked_map, tracked_players, frame_idx, label="player"
                )

            # -- Track ball --
            if track_ball and ball_tracker is not None and ball_mask.any():
                ball_detections = sv.Detections(
                    xyxy=all_boxes[ball_mask],
                    confidence=all_confs[ball_mask],
                    class_id=all_classes[ball_mask],
                )
                tracked_balls = ball_tracker.update_with_detections(
                    ball_detections
                )
                self._update_tracked_objects(
                    ball_tracked_map, tracked_balls, frame_idx, label="ball"
                )

            frame_idx += 1

        cap.release()

        if progress_callback:
            progress_callback("Tracking complete!")

        # ---- Build final object list ----
        # Assign globally unique IDs: players first, then ball
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
            label: Label for new objects ("player" or "ball").
        """
        if detections.tracker_id is None:
            return

        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)

            if tid not in tracked_map:
                tracked_map[tid] = TrackedObject(
                    object_id=tid,  # temporary, reassigned later
                    label=label,
                )

            obj = tracked_map[tid]
            obj.boxes[frame_idx] = detections.xyxy[i].copy()

            if detections.confidence is not None and i < len(detections.confidence):
                obj.scores[frame_idx] = float(detections.confidence[i])
