"""Ball speed estimation from tracking data.

Computes instantaneous and smoothed ball speed from frame-to-frame
centroid displacement, with optional pixel-to-meter calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeedEstimate:
    """Speed data for a single frame transition."""

    frame_index: int
    speed_px_per_frame: float | None  # raw pixel displacement per frame
    speed_kmh: float | None  # estimated km/h (None if no calibration)
    centroid: tuple[float, float] | None  # ball position at this frame


@dataclass
class SpeedAnalysis:
    """Full speed analysis for a tracked ball."""

    estimates: list[SpeedEstimate]
    avg_speed_kmh: float | None
    max_speed_kmh: float | None
    min_speed_kmh: float | None

    def get_speeds_kmh(self) -> list[float | None]:
        """Get list of speeds in km/h (one per frame)."""
        return [e.speed_kmh for e in self.estimates]

    def get_speeds_px(self) -> list[float | None]:
        """Get list of speeds in pixels/frame."""
        return [e.speed_px_per_frame for e in self.estimates]

    def get_frame_indices(self) -> list[int]:
        """Get frame indices."""
        return [e.frame_index for e in self.estimates]


def compute_ball_speed(
    centroids: dict[int, tuple[float, float] | None],
    fps: float,
    px_per_meter: float | None = None,
    smoothing_window: int = 5,
) -> SpeedAnalysis:
    """Compute ball speed from per-frame centroid positions.

    Args:
        centroids: Dict mapping frame_index -> (cx, cy) centroid, or None if
                   the ball was not detected in that frame.
        fps: Video frames per second.
        px_per_meter: Pixel-to-meter calibration factor. If None, km/h values
                      will be estimated with a default heuristic.
        smoothing_window: Window size for moving-average smoothing. Set to 1
                         for no smoothing.

    Returns:
        SpeedAnalysis with per-frame speed estimates and summary statistics.
    """
    if not centroids:
        return SpeedAnalysis(estimates=[], avg_speed_kmh=None,
                             max_speed_kmh=None, min_speed_kmh=None)

    sorted_frames = sorted(centroids.keys())
    estimates: list[SpeedEstimate] = []

    # First frame has no speed
    estimates.append(SpeedEstimate(
        frame_index=sorted_frames[0],
        speed_px_per_frame=None,
        speed_kmh=None,
        centroid=centroids.get(sorted_frames[0]),
    ))

    raw_speeds_px: list[float | None] = [None]

    for i in range(1, len(sorted_frames)):
        curr_frame = sorted_frames[i]
        prev_frame = sorted_frames[i - 1]
        curr_pos = centroids.get(curr_frame)
        prev_pos = centroids.get(prev_frame)

        if curr_pos is None or prev_pos is None:
            raw_speeds_px.append(None)
            estimates.append(SpeedEstimate(
                frame_index=curr_frame,
                speed_px_per_frame=None,
                speed_kmh=None,
                centroid=curr_pos,
            ))
            continue

        # Compute pixel displacement
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        frame_gap = curr_frame - prev_frame  # handle non-consecutive frames
        px_dist = np.sqrt(dx ** 2 + dy ** 2)
        px_per_frame = px_dist / max(frame_gap, 1)

        raw_speeds_px.append(px_per_frame)
        estimates.append(SpeedEstimate(
            frame_index=curr_frame,
            speed_px_per_frame=px_per_frame,
            speed_kmh=None,  # filled after smoothing
            centroid=curr_pos,
        ))

    # Apply smoothing
    smoothed = _smooth_speeds(raw_speeds_px, smoothing_window)

    # Convert to km/h
    valid_kmh: list[float] = []
    for i, est in enumerate(estimates):
        if smoothed[i] is not None:
            if px_per_meter is not None:
                meters_per_frame = smoothed[i] / px_per_meter
            else:
                # Default heuristic: assume typical broadcast frame is ~105m wide
                # (standard pitch length) and ~1920px wide
                meters_per_frame = smoothed[i] / _default_px_per_meter()

            speed_mps = meters_per_frame * fps
            speed_kmh = speed_mps * 3.6
            est.speed_kmh = round(speed_kmh, 1)
            est.speed_px_per_frame = smoothed[i]
            valid_kmh.append(speed_kmh)

    avg_kmh = round(np.mean(valid_kmh), 1) if valid_kmh else None
    max_kmh = round(max(valid_kmh), 1) if valid_kmh else None
    min_kmh = round(min(valid_kmh), 1) if valid_kmh else None

    return SpeedAnalysis(
        estimates=estimates,
        avg_speed_kmh=avg_kmh,
        max_speed_kmh=max_kmh,
        min_speed_kmh=min_kmh,
    )


def estimate_px_per_meter(
    known_distance_meters: float,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """Calculate pixels-per-meter from two known points on the field.

    For example, the penalty box width is 40.3 meters. If you can identify
    the pixel positions of the left and right edges of the penalty box,
    you can compute the calibration factor.

    Args:
        known_distance_meters: Real-world distance between the two points.
        point_a: (x, y) pixel position of the first point.
        point_b: (x, y) pixel position of the second point.

    Returns:
        Pixels per meter calibration factor.
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    px_dist = np.sqrt(dx ** 2 + dy ** 2)
    return px_dist / known_distance_meters


def _default_px_per_meter() -> float:
    """Default heuristic for px/meter on a standard broadcast view.

    Assumes a typical broadcast camera shows roughly the full pitch width
    (68m) across a 1920px-wide frame.
    """
    return 1920.0 / 68.0  # ~28.2 px/meter


def _smooth_speeds(
    speeds: list[float | None], window: int
) -> list[float | None]:
    """Apply moving average smoothing to speed values, skipping Nones."""
    if window <= 1:
        return speeds

    smoothed: list[float | None] = []
    for i in range(len(speeds)):
        if speeds[i] is None:
            smoothed.append(None)
            continue

        # Gather values within the window
        values = []
        half = window // 2
        for j in range(max(0, i - half), min(len(speeds), i + half + 1)):
            if speeds[j] is not None:
                values.append(speeds[j])

        smoothed.append(np.mean(values) if values else None)

    return smoothed
