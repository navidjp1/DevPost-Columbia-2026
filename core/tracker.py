"""SAM 3 video tracking wrapper for soccer match analysis.

Wraps the SAM 3 Video Predictor API to track players and ball
using text prompts across video frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Represents a single tracked object across all frames."""

    object_id: int
    label: str  # e.g. "team_a", "team_b", "ball"
    prompt: str  # the text prompt used to detect this object
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
    """High-level tracker that uses SAM 3 to track players and ball in soccer video.

    Usage:
        tracker = SoccerTracker()
        result = tracker.analyze(
            video_path="clip.mp4",
            team_a_prompt="player in white jersey",
            team_b_prompt="player in red jersey",
            track_ball=True,
        )
    """

    def __init__(self, device: str | None = None):
        """Initialize the tracker.

        Args:
            device: Torch device string. If None, auto-detects GPU.
        """
        self._predictor = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_predictor(self):
        """Lazily load the SAM 3 video predictor."""
        if self._predictor is not None:
            return

        logger.info("Loading SAM 3 video predictor...")
        from sam3.model_builder import build_sam3_video_predictor

        self._predictor = build_sam3_video_predictor()
        logger.info("SAM 3 video predictor loaded successfully.")

    def analyze(
        self,
        video_path: str,
        team_a_prompt: str = "player in white jersey",
        team_b_prompt: str = "player in red jersey",
        track_ball: bool = True,
        ball_prompt: str = "soccer ball",
        prompt_frame: int = 0,
        fps: float | None = None,
        progress_callback: Any | None = None,
    ) -> TrackingResult:
        """Run full tracking analysis on a soccer video clip.

        Args:
            video_path: Path to the video file (MP4) or JPEG frame directory.
            team_a_prompt: Text prompt for team A players.
            team_b_prompt: Text prompt for team B players.
            track_ball: Whether to track the ball.
            ball_prompt: Text prompt for the ball.
            prompt_frame: Frame index to use for initial detection.
            fps: Video FPS (auto-detected from video if None).
            progress_callback: Optional callable(status_str) for progress updates.

        Returns:
            TrackingResult with all tracked objects and their per-frame data.
        """
        import cv2

        self._load_predictor()

        # Get video metadata if fps not provided
        if fps is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        if progress_callback:
            progress_callback("Starting SAM 3 session...")

        # Start a video session
        response = self._predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        try:
            tracked_objects: list[TrackedObject] = []
            object_id_counter = 0

            # Build list of prompts to process
            prompts = [
                (team_a_prompt, "team_a"),
                (team_b_prompt, "team_b"),
            ]
            if track_ball:
                prompts.append((ball_prompt, "ball"))

            # Add prompts for each category
            for prompt_text, label in prompts:
                if progress_callback:
                    progress_callback(f"Detecting: {prompt_text}...")

                resp = self._predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=prompt_frame,
                        text=prompt_text,
                    )
                )

                outputs = resp.get("outputs", {})
                masks = outputs.get("masks")
                boxes = outputs.get("boxes")
                scores = outputs.get("scores")

                if masks is None or boxes is None:
                    logger.warning(
                        f"No detections for prompt '{prompt_text}' on frame {prompt_frame}"
                    )
                    continue

                # Convert to numpy if needed
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()

                # Create TrackedObject for each detected instance
                num_instances = len(boxes) if hasattr(boxes, '__len__') else 0
                for i in range(num_instances):
                    obj = TrackedObject(
                        object_id=object_id_counter,
                        label=label,
                        prompt=prompt_text,
                    )
                    if masks is not None and i < len(masks):
                        obj.masks[prompt_frame] = (
                            masks[i] if masks.ndim > 2 else masks
                        )
                    if boxes is not None and i < len(boxes):
                        obj.boxes[prompt_frame] = np.array(boxes[i])
                    if scores is not None and i < len(scores):
                        obj.scores[prompt_frame] = float(scores[i])

                    tracked_objects.append(obj)
                    object_id_counter += 1

            # Propagate tracking to all frames
            if progress_callback:
                progress_callback("Propagating tracking across all frames...")

            frame_indices = list(range(total_frames))
            if frame_indices:
                propagation_resp = self._predictor.handle_request(
                    request=dict(
                        type="get_outputs",
                        session_id=session_id,
                        frame_indices=frame_indices,
                    )
                )

                # Parse propagation outputs and assign to tracked objects
                self._parse_propagation_outputs(
                    propagation_resp, tracked_objects, frame_indices
                )

            if progress_callback:
                progress_callback("Tracking complete!")

            return TrackingResult(
                objects=tracked_objects,
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
            )

        finally:
            # Always clean up the session
            try:
                self._predictor.handle_request(
                    request=dict(
                        type="end_session",
                        session_id=session_id,
                    )
                )
            except Exception as e:
                logger.warning(f"Error ending SAM 3 session: {e}")

    def _parse_propagation_outputs(
        self,
        response: dict,
        tracked_objects: list[TrackedObject],
        frame_indices: list[int],
    ):
        """Parse propagation response and update tracked objects with per-frame data.

        The exact response format depends on the SAM 3 version. This method
        handles the common output structure where outputs contain masks, boxes,
        and scores indexed by frame.
        """
        outputs = response.get("outputs", response)

        # Handle different possible output formats from SAM 3
        # Format 1: outputs is a dict with frame_index keys
        if isinstance(outputs, dict):
            for frame_idx in frame_indices:
                frame_key = frame_idx
                if frame_key not in outputs and str(frame_key) in outputs:
                    frame_key = str(frame_key)
                if frame_key not in outputs:
                    continue

                frame_data = outputs[frame_key]
                masks = frame_data.get("masks")
                boxes = frame_data.get("boxes")
                scores = frame_data.get("scores")

                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()

                if boxes is not None:
                    for i, obj in enumerate(tracked_objects):
                        if i < len(boxes):
                            obj.boxes[frame_idx] = np.array(boxes[i])
                        if masks is not None and i < len(masks):
                            obj.masks[frame_idx] = (
                                masks[i] if masks.ndim > 2 else masks
                            )
                        if scores is not None and i < len(scores):
                            obj.scores[frame_idx] = float(scores[i])

        # Format 2: outputs is a list indexed by frame
        elif isinstance(outputs, (list, tuple)):
            for idx, frame_idx in enumerate(frame_indices):
                if idx >= len(outputs):
                    break
                frame_data = outputs[idx]
                if not isinstance(frame_data, dict):
                    continue

                masks = frame_data.get("masks")
                boxes = frame_data.get("boxes")
                scores = frame_data.get("scores")

                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()

                if boxes is not None:
                    for i, obj in enumerate(tracked_objects):
                        if i < len(boxes):
                            obj.boxes[frame_idx] = np.array(boxes[i])
                        if masks is not None and i < len(masks):
                            obj.masks[frame_idx] = (
                                masks[i] if masks.ndim > 2 else masks
                            )
                        if scores is not None and i < len(scores):
                            obj.scores[frame_idx] = float(scores[i])

    def track_single_player(
        self,
        video_path: str,
        frame_index: int,
        point: tuple[int, int] | None = None,
        box: list[int] | None = None,
        text: str | None = None,
        fps: float | None = None,
        progress_callback: Any | None = None,
    ) -> TrackingResult:
        """Track a single player using a visual or text prompt.

        Provide exactly one of: point, box, or text.

        Args:
            video_path: Path to the video file.
            frame_index: Frame to place the prompt on.
            point: (x, y) pixel coordinate to click on the player.
            box: [x1, y1, x2, y2] bounding box around the player.
            text: Text description of the player.
            fps: Video FPS (auto-detected if None).
            progress_callback: Optional callable(status_str).

        Returns:
            TrackingResult with a single tracked object.
        """
        import cv2

        self._load_predictor()

        cap = cv2.VideoCapture(video_path)
        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        fps = fps or detected_fps

        if progress_callback:
            progress_callback("Starting single-player tracking session...")

        response = self._predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        try:
            # Build prompt request
            prompt_request: dict[str, Any] = dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
            )

            if text is not None:
                prompt_request["text"] = text
            elif point is not None:
                prompt_request["points"] = [list(point)]
                prompt_request["labels"] = [1]  # positive label
            elif box is not None:
                prompt_request["box"] = box
            else:
                raise ValueError("Must provide one of: point, box, or text")

            if progress_callback:
                progress_callback("Detecting player...")

            resp = self._predictor.handle_request(request=prompt_request)

            outputs = resp.get("outputs", {})
            masks = outputs.get("masks")
            boxes = outputs.get("boxes")
            scores = outputs.get("scores")

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            # Take the best detection
            obj = TrackedObject(
                object_id=0,
                label="highlighted_player",
                prompt=text or (f"point({point})" if point else f"box({box})"),
            )
            if boxes is not None and len(boxes) > 0:
                obj.boxes[frame_index] = np.array(boxes[0])
            if masks is not None and len(masks) > 0:
                obj.masks[frame_index] = masks[0] if masks.ndim > 2 else masks
            if scores is not None and len(scores) > 0:
                obj.scores[frame_index] = float(scores[0])

            # Propagate across all frames
            if progress_callback:
                progress_callback("Propagating single-player tracking...")

            frame_indices = list(range(total_frames))
            propagation_resp = self._predictor.handle_request(
                request=dict(
                    type="get_outputs",
                    session_id=session_id,
                    frame_indices=frame_indices,
                )
            )
            self._parse_propagation_outputs(
                propagation_resp, [obj], frame_indices
            )

            if progress_callback:
                progress_callback("Single-player tracking complete!")

            return TrackingResult(
                objects=[obj],
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
            )

        finally:
            try:
                self._predictor.handle_request(
                    request=dict(type="end_session", session_id=session_id)
                )
            except Exception as e:
                logger.warning(f"Error ending SAM 3 session: {e}")
