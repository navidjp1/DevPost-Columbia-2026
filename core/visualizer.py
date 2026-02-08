"""Visualization engine for annotating soccer video frames.

Uses **supervision** annotators (EllipseAnnotator, TriangleAnnotator,
LabelAnnotator) for clean, game-broadcast-style annotations -- closely
matching the approach from:
  https://github.com/zakroum-hicham/football-analysis-CV

Also provides a combined video + pitch-diagram output and ball-possession
overlay drawing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import supervision as sv

from core.tracker import TrackedObject, TrackingResult
from core.speed import SpeedAnalysis

# -----------------------------------------------------------------------
# Colour palette (supervision Color objects)
# -----------------------------------------------------------------------

COLORS = {
    "team_a": sv.ColorPalette.from_hex(["#1E90FF"]),       # Dodger blue
    "team_b": sv.ColorPalette.from_hex(["#DC143C"]),       # Crimson
    "referee": sv.ColorPalette.from_hex(["#FFD700"]),      # Gold
    "goalkeeper": sv.ColorPalette.from_hex(["#FFFFFF"]),    # White
    "ball": sv.Color.from_hex("#FF8C00"),                   # Dark orange
    "active_player": sv.Color.from_rgb_tuple((255, 0, 0)), # Red
    "label_text": sv.Color.from_hex("#000000"),             # Black text
}

# BGR tuples for raw OpenCV drawing
_BGR_TEAM_A = (255, 144, 30)   # #1E90FF in BGR
_BGR_TEAM_B = (60, 20, 220)    # #DC143C in BGR
_BGR_BALL = (0, 140, 255)      # #FF8C00 in BGR
_BGR_REFEREE = (0, 215, 255)   # #FFD700 in BGR

# -----------------------------------------------------------------------
# Supervision annotator instances (created once, reused)
# -----------------------------------------------------------------------

LABEL_POS = sv.Position.BOTTOM_CENTER

team_a_ellipse = sv.EllipseAnnotator(color=COLORS["team_a"])
team_b_ellipse = sv.EllipseAnnotator(color=COLORS["team_b"])
referee_ellipse = sv.EllipseAnnotator(color=COLORS["referee"])
gk_ellipse = sv.EllipseAnnotator(color=COLORS["goalkeeper"])

ball_triangle = sv.TriangleAnnotator(color=COLORS["ball"], base=18, height=18)
active_triangle = sv.TriangleAnnotator(color=COLORS["active_player"], base=18, height=18)

team_a_label = sv.LabelAnnotator(color=COLORS["team_a"], text_color=COLORS["label_text"], text_position=LABEL_POS)
team_b_label = sv.LabelAnnotator(color=COLORS["team_b"], text_color=COLORS["label_text"], text_position=LABEL_POS)
referee_label = sv.LabelAnnotator(color=COLORS["referee"], text_color=COLORS["label_text"], text_position=LABEL_POS, border_radius=30)
gk_label = sv.LabelAnnotator(color=COLORS["goalkeeper"], text_color=COLORS["label_text"], text_position=LABEL_POS, border_radius=30)


# -----------------------------------------------------------------------
# Ball possession overlay
# -----------------------------------------------------------------------

def draw_ball_possession(
    frame: np.ndarray,
    ball_possession: dict[str, int],
) -> np.ndarray:
    """Draw a translucent ball-possession bar at the top of the frame."""
    total = ball_possession.get("team_a", 0) + ball_possession.get("team_b", 0)
    if total == 0:
        return frame

    pct_a = ball_possession["team_a"] / total
    pct_b = ball_possession["team_b"] / total

    overlay = frame.copy()
    h, w = frame.shape[:2]
    # Background bar
    cv2.rectangle(overlay, (w - 630, 60), (w - 10, 140), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Ball Control:", (w - 620, 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f"{pct_a * 100:.1f}%", (w - 360, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _BGR_TEAM_A, 2)
    cv2.putText(frame, f"{pct_b * 100:.1f}%", (w - 200, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _BGR_TEAM_B, 2)
    return frame


# -----------------------------------------------------------------------
# Single-frame annotation (supervision-based)
# -----------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    tracking_result: TrackingResult,
    frame_index: int,
    speed_analysis: SpeedAnalysis | None = None,
    highlighted_player_id: int | None = None,
    show_trails: bool = True,
    trail_length: int = 30,
    show_masks: bool = False,
    show_possession: bool = True,
) -> np.ndarray:
    """Annotate a single frame with supervision-style ellipses and labels.

    Args:
        frame: BGR numpy array.
        tracking_result: Full tracking result.
        frame_index: Current frame index.
        speed_analysis: Optional ball speed overlay data.
        highlighted_player_id: Player to highlight (unused for now).
        show_trails: Whether to draw ball trail.
        trail_length: Trail length in frames.
        show_masks: Unused (kept for API compat).
        show_possession: Whether to show ball possession overlay.

    Returns:
        Annotated BGR frame.
    """
    annotated = frame.copy()

    # Gather per-label detections for this frame
    ta_boxes, ta_ids = [], []
    tb_boxes, tb_ids = [], []
    ref_boxes = []
    ball_boxes = []

    for obj in tracking_result.objects:
        if frame_index not in obj.boxes:
            continue
        box = obj.boxes[frame_index]
        if obj.label == "team_a":
            ta_boxes.append(box)
            ta_ids.append(obj.object_id)
        elif obj.label == "team_b":
            tb_boxes.append(box)
            tb_ids.append(obj.object_id)
        elif obj.label == "referee":
            ref_boxes.append(box)
        elif obj.label == "ball":
            ball_boxes.append(box)

    # -- Team A --
    if ta_boxes:
        det = sv.Detections(xyxy=np.array(ta_boxes, dtype=np.float32))
        labels = [str(tid) for tid in ta_ids]
        annotated = team_a_ellipse.annotate(scene=annotated, detections=det)
        annotated = team_a_label.annotate(scene=annotated, detections=det, labels=labels)

    # -- Team B --
    if tb_boxes:
        det = sv.Detections(xyxy=np.array(tb_boxes, dtype=np.float32))
        labels = [str(tid) for tid in tb_ids]
        annotated = team_b_ellipse.annotate(scene=annotated, detections=det)
        annotated = team_b_label.annotate(scene=annotated, detections=det, labels=labels)

    # -- Referees --
    if ref_boxes:
        det = sv.Detections(xyxy=np.array(ref_boxes, dtype=np.float32))
        labels = ["ref"] * len(ref_boxes)
        annotated = referee_ellipse.annotate(scene=annotated, detections=det)
        annotated = referee_label.annotate(scene=annotated, detections=det, labels=labels)

    # -- Ball --
    if ball_boxes:
        padded = sv.pad_boxes(xyxy=np.array(ball_boxes, dtype=np.float32), px=10)
        det = sv.Detections(xyxy=padded)
        annotated = ball_triangle.annotate(scene=annotated, detections=det)

    # -- Ball trail --
    if show_trails and ball_boxes:
        ball_obj = tracking_result.get_ball()
        if ball_obj:
            _draw_trail(annotated, ball_obj, frame_index, trail_length, _BGR_BALL)

    # -- Ball possession --
    if show_possession and tracking_result.ball_possession:
        annotated = draw_ball_possession(annotated, tracking_result.ball_possession)

    # -- Speed overlay --
    if speed_analysis is not None:
        _draw_speed_overlay(annotated, speed_analysis, frame_index)

    return annotated


# -----------------------------------------------------------------------
# Batch annotation
# -----------------------------------------------------------------------

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
    """Annotate all frames in a video."""
    annotated_frames = []
    total = len(frames)
    for i, frame in enumerate(frames):
        if progress_callback and i % 30 == 0:
            progress_callback(f"Annotating frame {i + 1}/{total}...")
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


# -----------------------------------------------------------------------
# Combined video + pitch diagram (like the reference notebook)
# -----------------------------------------------------------------------

def annotate_frame_with_pitch(
    frame: np.ndarray,
    tracking_result: TrackingResult,
    frame_index: int,
    transformer,
    pitch_config=None,
    speed_analysis: SpeedAnalysis | None = None,
    show_trails: bool = True,
    trail_length: int = 30,
    base_pitch: np.ndarray | None = None,
) -> np.ndarray:
    """Annotate a frame and stack a pitch diagram below it.

    This replicates the combined output from the reference notebook.
    """
    from core.view_transformer import (
        SoccerPitchConfiguration,
        draw_pitch,
        draw_points_on_pitch,
        CONFIG,
    )

    cfg = pitch_config or CONFIG

    # Annotate the video frame
    annotated = annotate_frame(
        frame=frame,
        tracking_result=tracking_result,
        frame_index=frame_index,
        speed_analysis=speed_analysis,
        show_trails=show_trails,
        trail_length=trail_length,
    )

    # Draw pitch diagram
    pitch_img = draw_pitch(cfg) if base_pitch is None else base_pitch.copy()

    # Collect per-label positions and project to pitch
    for obj in tracking_result.objects:
        foot = obj.get_foot_position(frame_index)
        if foot is None:
            continue

        pts = np.array([[foot[0], foot[1]]], dtype=np.float32)
        pitch_xy = transformer.transform_points(pts)

        if obj.label == "team_a":
            face = (255, 191, 0)  # #00BFFF in BGR
        elif obj.label == "team_b":
            face = (147, 20, 255)  # #FF1493 in BGR
        elif obj.label == "referee":
            face = (0, 215, 255)  # #FFD700 in BGR
        elif obj.label == "ball":
            face = (255, 255, 255)
        else:
            continue

        pitch_img = draw_points_on_pitch(
            config=cfg,
            xy=pitch_xy,
            face_color=face,
            edge_color=(0, 0, 0),
            radius=16 if obj.label != "ball" else 10,
            pitch=pitch_img,
        )

    # Combine: video on top, pitch below (centred with black padding)
    vh, vw = annotated.shape[:2]
    ph, pw = pitch_img.shape[:2]

    pad_left = (vw - pw) // 2
    pad_right = vw - pw - pad_left

    if pad_left > 0 or pad_right > 0:
        padded_pitch = cv2.copyMakeBorder(
            pitch_img, 0, 0,
            max(0, pad_left), max(0, pad_right),
            cv2.BORDER_CONSTANT, value=[0, 0, 0],
        )
    else:
        # If pitch is wider than video, resize pitch to fit
        padded_pitch = cv2.resize(pitch_img, (vw, int(ph * vw / pw)))

    combined = np.vstack((annotated, padded_pitch))
    return combined


def annotate_all_frames_with_pitch(
    frames: list[np.ndarray],
    tracking_result: TrackingResult,
    transformer,
    pitch_config=None,
    speed_analysis: SpeedAnalysis | None = None,
    show_trails: bool = True,
    trail_length: int = 30,
    progress_callback=None,
) -> list[np.ndarray]:
    """Annotate all frames with combined video + pitch diagram.

    This is the main output function that replicates the reference notebook.
    """
    from core.view_transformer import draw_pitch, CONFIG

    cfg = pitch_config or CONFIG
    base = draw_pitch(cfg)

    results = []
    total = len(frames)
    for i, frame in enumerate(frames):
        if progress_callback and i % 30 == 0:
            progress_callback(f"Rendering combined frame {i + 1}/{total}...")
        combined = annotate_frame_with_pitch(
            frame=frame,
            tracking_result=tracking_result,
            frame_index=i,
            transformer=transformer,
            pitch_config=cfg,
            speed_analysis=speed_analysis,
            show_trails=show_trails,
            trail_length=trail_length,
            base_pitch=base,
        )
        results.append(combined)
    return results


# -----------------------------------------------------------------------
# Standalone top-down pitch rendering (no video frame, just the pitch)
# -----------------------------------------------------------------------

def render_topdown_frame(
    frame_index: int,
    tracking_result: TrackingResult,
    view_transformer,
    pitch_config=None,
    base_pitch: np.ndarray | None = None,
) -> np.ndarray:
    """Render a single top-down pitch frame with player/ball/ref dots."""
    from core.view_transformer import draw_pitch, CONFIG

    cfg = pitch_config or CONFIG
    canvas = base_pitch.copy() if base_pitch is not None else draw_pitch(cfg)

    from core.view_transformer import draw_points_on_pitch

    for obj in tracking_result.objects:
        foot = obj.get_foot_position(frame_index)
        if foot is None:
            continue

        pts = np.array([[foot[0], foot[1]]], dtype=np.float32)
        pitch_xy = view_transformer.transform_points(pts)

        if obj.label == "team_a":
            face = (255, 191, 0)
        elif obj.label == "team_b":
            face = (147, 20, 255)
        elif obj.label == "referee":
            face = (0, 215, 255)
        elif obj.label == "ball":
            face = (255, 255, 255)
        else:
            continue

        canvas = draw_points_on_pitch(
            config=cfg,
            xy=pitch_xy,
            face_color=face,
            edge_color=(0, 0, 0),
            radius=16 if obj.label != "ball" else 10,
            pitch=canvas,
        )

    return canvas


def render_topdown_video(
    tracking_result: TrackingResult,
    view_transformer,
    pitch_config=None,
    progress_callback=None,
) -> list[np.ndarray]:
    """Render a full top-down video."""
    from core.view_transformer import draw_pitch, CONFIG

    cfg = pitch_config or CONFIG
    base = draw_pitch(cfg)
    frames = []
    total = tracking_result.total_frames

    for f in range(total):
        if progress_callback and f % 30 == 0:
            progress_callback(f"Rendering top-down frame {f}/{total}...")
        frame = render_topdown_frame(
            frame_index=f,
            tracking_result=tracking_result,
            view_transformer=view_transformer,
            pitch_config=cfg,
            base_pitch=base,
        )
        frames.append(frame)

    return frames


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def _draw_trail(
    frame: np.ndarray,
    obj: TrackedObject,
    current_frame: int,
    trail_length: int,
    color: tuple[int, int, int],
):
    """Draw a movement trail using direct line segments (fast)."""
    points = []
    start_frame = max(0, current_frame - trail_length)

    for f in range(start_frame, current_frame + 1):
        centroid = obj.get_centroid(f)
        if centroid is not None:
            points.append((int(centroid[0]), int(centroid[1])))

    if len(points) < 2:
        return

    n = len(points)
    for i in range(1, n):
        frac = i / n
        thickness = max(1, int(3 * frac))
        seg_color = (int(color[0] * frac), int(color[1] * frac), int(color[2] * frac))
        cv2.line(frame, points[i - 1], points[i], seg_color, thickness)


def _draw_speed_overlay(
    frame: np.ndarray,
    speed_analysis: SpeedAnalysis,
    frame_index: int,
):
    """Draw ball speed info on the frame."""
    speed_kmh = None
    for est in speed_analysis.estimates:
        if est.frame_index == frame_index:
            speed_kmh = est.speed_kmh
            break

    if speed_kmh is None:
        return

    h, w = frame.shape[:2]
    text = f"Ball Speed: {speed_kmh:.1f} km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 0.7, 2)
    x, y = w - tw - 20, 40

    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + baseline + 10), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + baseline + 10), _BGR_BALL, 2)
    cv2.putText(frame, text, (x, y), font, 0.7, _BGR_BALL, 2)


# Kept for backwards compatibility
def get_color_for_label(label: str) -> tuple[int, int, int]:
    """Get BGR color for an object label."""
    return {
        "team_a": _BGR_TEAM_A,
        "team_b": _BGR_TEAM_B,
        "ball": _BGR_BALL,
        "referee": _BGR_REFEREE,
    }.get(label, (200, 200, 200))
