"""Visualization engine for annotating soccer video frames.

Uses OpenCV and supervision to draw bounding boxes, trails,
labels, and speed overlays on video frames.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False

from core.tracker import TrackedObject, TrackingResult
from core.speed import SpeedAnalysis


# Color palette (BGR format for OpenCV)
PLAYER_COLOR = (255, 165, 0)     # Orange for all players
BALL_COLOR = (0, 255, 0)         # Green
HIGHLIGHT_COLOR = (0, 255, 255)  # Yellow
TRAIL_ALPHA = 0.6
DEFAULT_THICKNESS = 2
HIGHLIGHT_THICKNESS = 4


def get_color_for_label(label: str) -> tuple[int, int, int]:
    """Get BGR color for an object label."""
    colors = {
        "player": PLAYER_COLOR,
        "ball": BALL_COLOR,
        "highlighted_player": HIGHLIGHT_COLOR,
    }
    return colors.get(label, (200, 200, 200))


def annotate_frame(
    frame: np.ndarray,
    tracking_result: TrackingResult,
    frame_index: int,
    speed_analysis: SpeedAnalysis | None = None,
    highlighted_player_id: int | None = None,
    show_trails: bool = True,
    trail_length: int = 30,
    show_masks: bool = False,
) -> np.ndarray:
    """Annotate a single frame with tracking data.

    Args:
        frame: BGR numpy array of the frame.
        tracking_result: Full tracking result with all objects.
        frame_index: Current frame index.
        speed_analysis: Optional ball speed data to overlay.
        highlighted_player_id: If set, highlight this player.
        show_trails: Whether to draw movement trails.
        trail_length: Number of past frames to include in trails.
        show_masks: Whether to overlay segmentation masks.

    Returns:
        Annotated BGR frame.
    """
    annotated = frame.copy()

    for obj in tracking_result.objects:
        if frame_index not in obj.boxes:
            continue

        box = obj.boxes[frame_index]
        color = get_color_for_label(obj.label)
        is_highlighted = (
            highlighted_player_id is not None
            and obj.object_id == highlighted_player_id
        )
        thickness = HIGHLIGHT_THICKNESS if is_highlighted else DEFAULT_THICKNESS

        if is_highlighted:
            color = HIGHLIGHT_COLOR

        # Draw mask overlay if available and requested
        if show_masks and frame_index in obj.masks:
            mask = obj.masks[frame_index]
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                overlay = annotated.copy()
                overlay[mask > 0] = color
                cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label_text = _build_label(obj, is_highlighted)
        _draw_label(annotated, label_text, (x1, y1 - 10), color)

        # Draw trail for highlighted player or ball
        if show_trails and (is_highlighted or obj.label == "ball"):
            _draw_trail(
                annotated, obj, frame_index, trail_length, color
            )

    # Draw ball speed overlay
    if speed_analysis is not None:
        _draw_speed_overlay(annotated, speed_analysis, frame_index)

    return annotated


def annotate_all_frames(
    frames: list[np.ndarray],
    tracking_result: TrackingResult,
    speed_analysis: SpeedAnalysis | None = None,
    highlighted_player_id: int | None = None,
    show_trails: bool = True,
    trail_length: int = 30,
    show_masks: bool = False,
    progress_callback=None,
) -> list[np.ndarray]:
    """Annotate all frames in a video.

    Args:
        frames: List of BGR numpy arrays.
        tracking_result: Full tracking result.
        speed_analysis: Optional ball speed data.
        highlighted_player_id: Player ID to highlight.
        show_trails: Whether to draw movement trails.
        trail_length: Trail length in frames.
        show_masks: Whether to overlay segmentation masks.
        progress_callback: Optional callable(status_str).

    Returns:
        List of annotated BGR frames.
    """
    annotated_frames = []
    total = len(frames)

    for i, frame in enumerate(frames):
        if progress_callback and i % 30 == 0:
            progress_callback(
                f"Annotating frame {i + 1}/{total}..."
            )

        annotated = annotate_frame(
            frame=frame,
            tracking_result=tracking_result,
            frame_index=i,
            speed_analysis=speed_analysis,
            highlighted_player_id=highlighted_player_id,
            show_trails=show_trails,
            trail_length=trail_length,
            show_masks=show_masks,
        )
        annotated_frames.append(annotated)

    return annotated_frames


def create_player_highlight_frame(
    frame: np.ndarray,
    obj: TrackedObject,
    frame_index: int,
    trail_length: int = 60,
) -> np.ndarray:
    """Create a frame focused on a single player with their trail.

    Draws a prominent box, trail, and label for the given player.
    Dims the rest of the frame slightly.

    Args:
        frame: BGR numpy array.
        obj: The TrackedObject to highlight.
        frame_index: Current frame index.
        trail_length: How many past frames to show in the trail.

    Returns:
        Annotated frame with player highlight.
    """
    # Dim the background slightly
    annotated = cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0)

    if frame_index not in obj.boxes:
        return annotated

    box = obj.boxes[frame_index]
    x1, y1, x2, y2 = map(int, box)

    # Restore original brightness in player region (with padding)
    pad = 20
    ry1 = max(0, y1 - pad)
    ry2 = min(frame.shape[0], y2 + pad)
    rx1 = max(0, x1 - pad)
    rx2 = min(frame.shape[1], x2 + pad)
    annotated[ry1:ry2, rx1:rx2] = frame[ry1:ry2, rx1:rx2]

    # Draw highlight box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), HIGHLIGHT_COLOR, HIGHLIGHT_THICKNESS)

    # Draw trail
    _draw_trail(annotated, obj, frame_index, trail_length, HIGHLIGHT_COLOR)

    # Label
    label = f"Player #{obj.object_id}"
    _draw_label(annotated, label, (x1, y1 - 15), HIGHLIGHT_COLOR)

    return annotated


def _build_label(obj: TrackedObject, is_highlighted: bool) -> str:
    """Build the label string for an object."""
    if obj.label == "ball":
        return "Ball"

    prefix = ">>> " if is_highlighted else ""
    return f"{prefix}#{obj.object_id}"


def _draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
):
    """Draw a text label with a background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    y = max(y, h + 5)  # don't draw above the frame

    # Background rectangle
    cv2.rectangle(
        frame,
        (x, y - h - 5),
        (x + w + 5, y + baseline),
        color,
        cv2.FILLED,
    )

    # Text (white on colored background)
    cv2.putText(
        frame, text, (x + 2, y - 2), font, font_scale, (255, 255, 255), thickness
    )


def _draw_trail(
    frame: np.ndarray,
    obj: TrackedObject,
    current_frame: int,
    trail_length: int,
    color: tuple[int, int, int],
):
    """Draw a movement trail for an object."""
    points = []
    start_frame = max(0, current_frame - trail_length)

    for f in range(start_frame, current_frame + 1):
        centroid = obj.get_centroid(f)
        if centroid is not None:
            points.append((int(centroid[0]), int(centroid[1])))

    if len(points) < 2:
        return

    # Draw trail with fading opacity
    for i in range(1, len(points)):
        alpha = i / len(points)  # 0.0 (oldest) to 1.0 (newest)
        thickness = max(1, int(3 * alpha))

        # Blend color with alpha
        overlay = frame.copy()
        cv2.line(overlay, points[i - 1], points[i], color, thickness)
        cv2.addWeighted(overlay, alpha * TRAIL_ALPHA, frame, 1 - alpha * TRAIL_ALPHA, 0, frame)


def _draw_speed_overlay(
    frame: np.ndarray,
    speed_analysis: SpeedAnalysis,
    frame_index: int,
):
    """Draw ball speed information on the frame."""
    # Find the speed estimate for this frame
    speed_kmh = None
    for est in speed_analysis.estimates:
        if est.frame_index == frame_index:
            speed_kmh = est.speed_kmh
            break

    if speed_kmh is None:
        return

    # Draw speed in top-right corner
    h, w = frame.shape[:2]
    text = f"Ball Speed: {speed_kmh:.1f} km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - tw - 20
    y = 40

    # Background
    cv2.rectangle(
        frame, (x - 10, y - th - 10), (x + tw + 10, y + baseline + 10),
        (0, 0, 0), cv2.FILLED
    )
    cv2.rectangle(
        frame, (x - 10, y - th - 10), (x + tw + 10, y + baseline + 10),
        BALL_COLOR, 2
    )

    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, BALL_COLOR, thickness)
