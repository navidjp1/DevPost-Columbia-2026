"""Perspective transformation for top-down pitch view.

Supports two modes:
  1. **Automatic**: Uses the Roboflow ``football-field-detection-f07vi/14``
     keypoint model to detect 32 pitch landmarks per frame, then computes
     a homography (``cv2.findHomography``) to map pixels to real-world
     coordinates.  This is the approach from
     https://github.com/zakroum-hicham/football-analysis-CV.
  2. **Manual**: 4 user-supplied point correspondences with
     ``cv2.getPerspectiveTransform``.

Also provides pitch drawing and configuration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv


# =========================================================================
# Pitch configuration -- 32 keypoint layout (centimeters, matches Roboflow
# football-field-detection model output labels)
# =========================================================================

@dataclass
class SoccerPitchConfiguration:
    """Pitch layout with 32 labelled vertices (all values in cm).

    Vertex numbering matches the Roboflow football-field-detection model.
    """
    width: int = 7000   # sideline to sideline (cm)
    length: int = 12000  # goal-line to goal-line (cm)
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """Return the 32 pitch vertices as (x, y) in cm."""
        return [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) // 2),  # 2
            (0, (self.width - self.goal_box_width) // 2),  # 3
            (0, (self.width + self.goal_box_width) // 2),  # 4
            (0, (self.width + self.penalty_box_width) // 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) // 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) // 2),  # 8
            (self.penalty_spot_distance, self.width // 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) // 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) // 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) // 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) // 2),  # 13
            (self.length // 2, 0),  # 14
            (self.length // 2, self.width // 2 - self.centre_circle_radius),  # 15
            (self.length // 2, self.width // 2 + self.centre_circle_radius),  # 16
            (self.length // 2, self.width),  # 17
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) // 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) // 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) // 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) // 2),  # 21
            (self.length - self.penalty_spot_distance, self.width // 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) // 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) // 2),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) // 2),  # 26
            (self.length, (self.width - self.goal_box_width) // 2),  # 27
            (self.length, (self.width + self.goal_box_width) // 2),  # 28
            (self.length, (self.width + self.penalty_box_width) // 2),  # 29
            (self.length, self.width),  # 30
            (self.length // 2 - self.centre_circle_radius, self.width // 2),  # 31
            (self.length // 2 + self.centre_circle_radius, self.width // 2),  # 32
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19",
    ])


CONFIG = SoccerPitchConfiguration()


# =========================================================================
# Pitch drawing (matches the reference notebook)
# =========================================================================

def draw_pitch(
    config: SoccerPitchConfiguration | None = None,
    background_color: Tuple[int, int, int] = (34, 139, 34),
    line_color: Tuple[int, int, int] = (255, 255, 255),
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1,
) -> np.ndarray:
    """Draw a 2-D soccer pitch using the 32-vertex layout.

    Args:
        config: Pitch configuration (defaults to FIFA standard).
        background_color: BGR background colour.
        line_color: BGR line colour.
        padding: Pixel padding around the pitch.
        line_thickness: Thickness of pitch lines.
        point_radius: Radius for penalty spot dots.
        scale: Scaling factor (pitch coords are in cm).

    Returns:
        BGR image of the pitch.
    """
    if config is None:
        config = CONFIG

    sw = int(config.width * scale)
    sl = int(config.length * scale)
    scr = int(config.centre_circle_radius * scale)
    spsd = int(config.penalty_spot_distance * scale)

    pitch = np.ones(
        (sw + 2 * padding, sl + 2 * padding, 3), dtype=np.uint8
    ) * np.array(background_color, dtype=np.uint8)

    # Draw edges
    for start, end in config.edges:
        p1 = (int(config.vertices[start - 1][0] * scale) + padding,
              int(config.vertices[start - 1][1] * scale) + padding)
        p2 = (int(config.vertices[end - 1][0] * scale) + padding,
              int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(pitch, p1, p2, line_color, line_thickness)

    # Centre circle
    centre = (sl // 2 + padding, sw // 2 + padding)
    cv2.circle(pitch, centre, scr, line_color, line_thickness)

    # Penalty spots
    for spot in [
        (spsd + padding, sw // 2 + padding),
        (sl - spsd + padding, sw // 2 + padding),
    ]:
        cv2.circle(pitch, spot, point_radius, line_color, -1)

    return pitch


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: Tuple[int, int, int] = (0, 0, 255),
    edge_color: Tuple[int, int, int] = (0, 0, 0),
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw coloured dots on a pitch image.

    Args:
        xy: Nx2 array of pitch coordinates (in cm).
        face_color / edge_color: BGR colours.
        pitch: Existing pitch image (drawn if None).

    Returns:
        Pitch image with dots drawn.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for point in xy:
        sp = (int(point[0] * scale) + padding, int(point[1] * scale) + padding)
        cv2.circle(pitch, sp, radius, face_color, -1)
        cv2.circle(pitch, sp, radius, edge_color, thickness)

    return pitch


# =========================================================================
# ViewTransformer (supports both findHomography and getPerspectiveTransform)
# =========================================================================

class ViewTransformer:
    """Map pixel coordinates to real-world pitch coordinates.

    Accepts N >= 4 point correspondences and uses ``cv2.findHomography``
    (robust to outliers).  When exactly 4 points are given, falls back to
    ``cv2.getPerspectiveTransform`` for an exact solution.
    """

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ):
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        if source.shape != target.shape:
            raise ValueError("source and target must have the same shape.")
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("Points must be Nx2 arrays.")

        if len(source) == 4:
            self.m = cv2.getPerspectiveTransform(source, target)
        else:
            self.m, _ = cv2.findHomography(source, target)

        if self.m is None:
            raise ValueError("Could not compute homography matrix.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an Nx2 array of pixel coords to pitch coords."""
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2).astype(np.float32)

    def transform_point(self, px_x: float, px_y: float) -> Tuple[float, float] | None:
        """Transform a single pixel coord; returns None if invalid."""
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.m)
        return (float(out[0, 0, 0]), float(out[0, 0, 1]))


# =========================================================================
# Automatic field keypoint detection (Roboflow model)
# =========================================================================

class FieldKeypointDetector:
    """Detects pitch keypoints using the Roboflow field detection model.

    Creates a per-frame ViewTransformer for accurate perspective mapping.

    Usage::

        detector = FieldKeypointDetector(api_key="...")
        transformer = detector.detect(frame)
        if transformer:
            pitch_xy = transformer.transform_points(pixel_xy)
    """

    MODEL_ID = "football-field-detection-f07vi/14"

    def __init__(self, api_key: str, conf: float = 0.3, min_confidence: float = 0.5):
        import os
        os.environ["ROBOFLOW_API_KEY"] = api_key
        from inference import get_model
        self._model = get_model(model_id=self.MODEL_ID)
        self._conf = conf
        self._min_confidence = min_confidence
        self._config = CONFIG

    def detect(self, frame: np.ndarray) -> ViewTransformer | None:
        """Detect field keypoints in a frame and return a transformer.

        Returns None if not enough keypoints are detected.
        """
        result = self._model.infer(frame, confidence=self._conf)
        if not result:
            return None

        key_points = sv.KeyPoints.from_inference(result[0])
        if key_points.confidence is None or len(key_points.confidence) == 0:
            return None

        mask = key_points.confidence[0] > self._min_confidence
        if mask.sum() < 4:
            return None

        frame_pts = key_points.xy[0][mask]
        pitch_pts = np.array(self._config.vertices, dtype=np.float32)[mask]

        try:
            return ViewTransformer(source=frame_pts, target=pitch_pts)
        except ValueError:
            return None


# =========================================================================
# Manual presets (fallback when field detection is not available)
# =========================================================================

PRESETS: dict[str, dict] = {
    "broadcast_left_half": {
        "description": (
            "Standard broadcast camera showing the left half of the pitch."
        ),
        "source_points": np.array([
            [265, 275], [910, 260], [1640, 915], [110, 1035],
        ], dtype=np.float32),
        "target_points": np.array([
            [0, 0], [6000, 0], [6000, 7000], [0, 7000],
        ], dtype=np.float32),
    },
    "broadcast_full_pitch": {
        "description": "Wide broadcast view showing the full pitch.",
        "source_points": np.array([
            [100, 200], [1820, 200], [1700, 950], [220, 950],
        ], dtype=np.float32),
        "target_points": np.array([
            [0, 0], [12000, 0], [12000, 7000], [0, 7000],
        ], dtype=np.float32),
    },
}


def get_preset_names() -> list[str]:
    return list(PRESETS.keys())


def create_transformer_from_preset(preset_name: str) -> ViewTransformer:
    preset = PRESETS[preset_name]
    return ViewTransformer(
        source=preset["source_points"],
        target=preset["target_points"],
    )
