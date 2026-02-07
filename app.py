"""Soccer Match Analysis -- Streamlit Application.

Upload a short soccer clip and analyze it using YOLOv8/Roboflow + ByteTrack
to track players (with automatic team classification), the ball, compute
ball speed, and highlight individual players.

Run with: streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Soccer Match Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from utils.video import (
    get_video_metadata,
    extract_frames,
    frames_to_video,
    save_uploaded_video,
)
from core.tracker import SoccerTracker, TrackingResult
from core.speed import compute_ball_speed, SpeedAnalysis
from core.visualizer import annotate_all_frames


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "tracking_result": None,
        "speed_analysis": None,
        "annotated_video_path": None,
        "video_path": None,
        "video_metadata": None,
        "analysis_done": False,
        "highlighted_player_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar -- Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚽ Match Analysis Config")

st.sidebar.header("Roboflow Integration")
use_roboflow = st.sidebar.checkbox(
    "Use Roboflow football model (recommended, runs locally)", value=True
)
roboflow_api_key = None
roboflow_model_id = "football-players-detection-3zvbc/11"
if use_roboflow:
    roboflow_api_key = st.sidebar.text_input(
        "Roboflow API Key",
        type="password",
        help="Get a free API key at roboflow.com. Needed once to download the model; inference runs locally.",
    )
    if not roboflow_api_key:
        st.sidebar.warning("Enter your Roboflow API key (needed once to download model weights).")

st.sidebar.header("Detection Settings")
model_choice = st.sidebar.selectbox(
    "YOLO Model (used when Roboflow is off)",
    ["yolov8n.pt (fast)", "yolov8s.pt (balanced)", "yolov8m.pt (accurate)"],
    index=0,
)
model_name = model_choice.split(" ")[0]

confidence = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.3, 0.05)

st.sidebar.header("Ball Tracking")
track_ball = st.sidebar.checkbox("Track the ball", value=True)

st.sidebar.header("Display Options")
show_trails = st.sidebar.checkbox("Show movement trails", value=True)
show_masks = st.sidebar.checkbox("Show segmentation masks", value=False)
trail_length = st.sidebar.slider("Trail length (frames)", 10, 120, 30)

st.sidebar.header("Speed Calibration")
calibration_mode = st.sidebar.selectbox(
    "Calibration method",
    ["Auto (heuristic)", "Manual (known distance)"],
)
px_per_meter = None
if calibration_mode == "Manual (known distance)":
    st.sidebar.caption(
        "Provide a known real-world distance visible in the frame "
        "(e.g. penalty box width = 40.3m) and the approximate pixel "
        "distance between those two points."
    )
    known_meters = st.sidebar.number_input("Known distance (meters)", value=40.3)
    known_pixels = st.sidebar.number_input("Pixel distance", value=500)
    if known_pixels > 0:
        px_per_meter = known_pixels / known_meters


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("⚽ Soccer Match Analyzer")
st.markdown(
    "Upload a short soccer clip to automatically **detect and track players "
    "by team**, track the ball, and compute ball speed. Uses a purpose-built "
    "football detection model (runs locally) with jersey color clustering "
    "for automatic team classification."
)

# ---------------------------------------------------------------------------
# Step 1: Upload video
# ---------------------------------------------------------------------------
st.header("1. Upload Video Clip")

uploaded_file = st.file_uploader(
    "Choose a video file (MP4 recommended, keep it under 30 seconds for fast results)",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is not None:
    video_path = save_uploaded_video(uploaded_file)
    st.session_state["video_path"] = video_path

    try:
        meta = get_video_metadata(video_path)
        st.session_state["video_metadata"] = meta
    except Exception as e:
        st.error(f"Error reading video: {e}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resolution", f"{meta.width}x{meta.height}")
    col2.metric("FPS", f"{meta.fps:.1f}")
    col3.metric("Frames", str(meta.total_frames))
    col4.metric("Duration", f"{meta.duration_seconds:.1f}s")

    st.video(video_path)

# ---------------------------------------------------------------------------
# Step 2: Run Analysis
# ---------------------------------------------------------------------------
if st.session_state["video_path"] is not None:
    st.header("2. Run Analysis")

    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        video_path = st.session_state["video_path"]
        meta = st.session_state["video_metadata"]

        # Reset previous results
        st.session_state["analysis_done"] = False
        st.session_state["tracking_result"] = None
        st.session_state["speed_analysis"] = None
        st.session_state["annotated_video_path"] = None
        st.session_state["highlighted_player_id"] = None

        progress = st.progress(0, text="Initializing...")
        status_text = st.empty()

        def update_status(msg: str):
            status_text.text(msg)

        try:
            # -- Track players and ball --
            progress.progress(10, text="Loading detection model...")
            tracker = SoccerTracker(
                model_name=model_name,
                confidence=confidence,
            )

            progress.progress(20, text="Running detection, tracking, and team classification...")
            tracking_result: TrackingResult = tracker.analyze(
                video_path=video_path,
                track_ball=track_ball,
                fps=meta.fps,
                roboflow_api_key=roboflow_api_key if use_roboflow else None,
                roboflow_model_id=roboflow_model_id,
                progress_callback=update_status,
            )
            st.session_state["tracking_result"] = tracking_result

            # -- Compute ball speed --
            speed_analysis = None
            if track_ball:
                progress.progress(60, text="Computing ball speed...")
                ball = tracking_result.get_ball()
                if ball is not None:
                    centroids = {}
                    for f in range(tracking_result.total_frames):
                        centroids[f] = ball.get_centroid(f)
                    speed_analysis = compute_ball_speed(
                        centroids=centroids,
                        fps=meta.fps,
                        px_per_meter=px_per_meter,
                    )
                    st.session_state["speed_analysis"] = speed_analysis

            # -- Render annotated video --
            progress.progress(70, text="Extracting frames for annotation...")
            frames = extract_frames(video_path)

            progress.progress(80, text="Annotating frames...")
            annotated_frames = annotate_all_frames(
                frames=frames,
                tracking_result=tracking_result,
                speed_analysis=speed_analysis,
                highlighted_player_id=None,
                show_trails=show_trails,
                trail_length=trail_length,
                show_masks=show_masks,
                progress_callback=update_status,
            )

            progress.progress(90, text="Encoding output video...")
            output_path = os.path.join("outputs", "annotated_output.mp4")
            os.makedirs("outputs", exist_ok=True)
            frames_to_video(annotated_frames, output_path, meta.fps)
            st.session_state["annotated_video_path"] = output_path

            progress.progress(100, text="Analysis complete!")
            st.session_state["analysis_done"] = True
            status_text.empty()

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback
            st.expander("Error details").code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Step 3: Results
# ---------------------------------------------------------------------------
if st.session_state["analysis_done"]:
    st.header("3. Results")

    tracking_result: TrackingResult = st.session_state["tracking_result"]
    speed_analysis: SpeedAnalysis | None = st.session_state["speed_analysis"]

    # -- Summary metrics --
    team_a = tracking_result.get_objects_by_label("team_a")
    team_b = tracking_result.get_objects_by_label("team_b")
    ball = tracking_result.get_ball()

    col1, col2, col3 = st.columns(3)
    col1.metric("Team A Players", len(team_a))
    col2.metric("Team B Players", len(team_b))
    col3.metric("Ball Detected", "Yes" if ball else "No")

    # -- Annotated video --
    st.subheader("Annotated Video")
    if st.session_state["annotated_video_path"]:
        st.video(st.session_state["annotated_video_path"])

    # -- Ball speed chart --
    if speed_analysis is not None:
        st.subheader("Ball Speed Over Time")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Speed", f"{speed_analysis.avg_speed_kmh} km/h" if speed_analysis.avg_speed_kmh else "N/A")
        col2.metric("Max Speed", f"{speed_analysis.max_speed_kmh} km/h" if speed_analysis.max_speed_kmh else "N/A")
        col3.metric("Min Speed", f"{speed_analysis.min_speed_kmh} km/h" if speed_analysis.min_speed_kmh else "N/A")

        # Build chart data
        chart_frames = speed_analysis.get_frame_indices()
        chart_speeds = speed_analysis.get_speeds_kmh()

        # Filter out None values for charting
        valid_data = [
            (f, s) for f, s in zip(chart_frames, chart_speeds) if s is not None
        ]
        if valid_data:
            import pandas as pd

            df = pd.DataFrame(valid_data, columns=["Frame", "Speed (km/h)"])
            # Convert frame to time in seconds
            fps = tracking_result.fps
            df["Time (s)"] = df["Frame"] / fps
            st.line_chart(df, x="Time (s)", y="Speed (km/h)")

    # -- Player list & single-player highlight --
    st.subheader("Player Tracking")

    player_ids = tracking_result.get_player_ids()
    if player_ids:
        player_options = {
            pid: f"Player #{pid} ({tracking_result.get_object_by_id(pid).label.replace('_', ' ').title()})"
            for pid in player_ids
        }

        selected_label = st.selectbox(
            "Select a player to highlight",
            options=["None"] + list(player_options.values()),
        )

        if selected_label != "None":
            # Find the ID from the label
            selected_id = None
            for pid, label in player_options.items():
                if label == selected_label:
                    selected_id = pid
                    break

            if selected_id is not None and selected_id != st.session_state.get("highlighted_player_id"):
                st.session_state["highlighted_player_id"] = selected_id

                with st.spinner("Re-rendering with player highlight..."):
                    video_path = st.session_state["video_path"]
                    meta = st.session_state["video_metadata"]
                    frames = extract_frames(video_path)

                    annotated_frames = annotate_all_frames(
                        frames=frames,
                        tracking_result=tracking_result,
                        speed_analysis=speed_analysis,
                        highlighted_player_id=selected_id,
                        show_trails=show_trails,
                        trail_length=trail_length,
                        show_masks=show_masks,
                    )

                    highlight_path = os.path.join("outputs", f"highlight_player_{selected_id}.mp4")
                    frames_to_video(annotated_frames, highlight_path, meta.fps)

                    st.video(highlight_path)
        else:
            st.session_state["highlighted_player_id"] = None

    # -- Detected objects table --
    with st.expander("All Detected Objects"):
        for obj in tracking_result.objects:
            frames_detected = len(obj.boxes)
            pct = (frames_detected / tracking_result.total_frames * 100) if tracking_result.total_frames > 0 else 0
            label_display = obj.label.replace("_", " ").title()
            st.write(
                f"**{label_display}** #{obj.object_id} "
                f"-- Detected in {frames_detected}/{tracking_result.total_frames} frames "
                f"({pct:.0f}%)"
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Roboflow Inference (local) + ByteTrack | Streamlit | OpenCV | "
    "Columbia DevPost Hackathon 2026"
)
