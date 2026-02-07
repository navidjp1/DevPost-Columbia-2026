"""Video I/O utilities for frame extraction and video reconstruction."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Container for video metadata."""

    fps: float
    width: int
    height: int
    total_frames: int
    duration_seconds: float


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata with fps, dimensions, frame count, and duration.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_seconds=duration,
        )
    finally:
        cap.release()


def extract_frames(video_path: str, max_frames: int | None = None) -> list[np.ndarray]:
    """Extract all frames from a video as BGR numpy arrays.

    Args:
        video_path: Path to the video file.
        max_frames: Optional limit on number of frames to extract.

    Returns:
        List of BGR numpy arrays (one per frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    return frames


def extract_frames_to_dir(video_path: str, output_dir: str | None = None,
                          max_frames: int | None = None) -> tuple[str, int]:
    """Extract frames from a video and save as JPEG files in a directory.

    SAM 3 can accept a JPEG folder as input, so this is useful for
    preparing video input.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save frames. If None, creates a temp dir.
        max_frames: Optional limit on number of frames to extract.

    Returns:
        Tuple of (output directory path, number of frames extracted).
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="soccer_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # SAM 3 expects JPEG files named as zero-padded numbers
            frame_path = os.path.join(output_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()

    return output_dir, frame_count


def frames_to_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float,
    codec: str = "mp4v",
) -> str:
    """Write a list of frames to a video file.

    Args:
        frames: List of BGR numpy arrays (all same size).
        output_path: Path for the output video file.
        fps: Frames per second for the output video.
        codec: FourCC codec string (default "mp4v" for .mp4 files).

    Returns:
        The output path.
    """
    if not frames:
        raise ValueError("No frames to write")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def save_uploaded_video(uploaded_file, output_dir: str = "uploads") -> str:
    """Save a Streamlit UploadedFile to disk and return the path.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        output_dir: Directory to save the file to.

    Returns:
        Path to the saved video file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, uploaded_file.name)

    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return output_path
