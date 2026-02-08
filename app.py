"""Soccer Match Analysis -- Streamlit Application.

Upload a short soccer clip and analyze it using:
  - A custom-trained YOLOv8 model (trained via training/train.py), OR
  - The Roboflow football-players-detection model (local inference).

Detection → ByteTrack → KMeans team assignment → Ball possession →
(optional) Roboflow field keypoint detection for automatic top-down view.

Based on: https://github.com/zakroum-hicham/football-analysis-CV

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
from core.visualizer import (
    annotate_all_frames,
    annotate_all_frames_with_pitch,
    render_topdown_video,
)
from core.view_transformer import (
    create_transformer_from_preset,
    get_preset_names,
    FieldKeypointDetector,
)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "tracking_result": None,
        "speed_analysis": None,
        "annotated_video_path": None,
        "topdown_video_path": None,
        "combined_video_path": None,
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

# ---- Model source ----
st.sidebar.header("Detection Model")
model_source = st.sidebar.radio(
    "Model source",
    ["Custom trained model (best.pt)", "Roboflow API (local inference)", "Generic YOLO"],
    index=1,
    help=(
        "**Custom trained**: Use a model you trained with `training/train.py`. "
        "**Roboflow API**: Download & run the football-players-detection model locally. "
        "**Generic YOLO**: Use a pre-trained YOLOv8 model (less accurate for football)."
    ),
)

custom_model_path = None
roboflow_api_key = None
roboflow_model_id = "football-players-detection-3zvbc/11"
model_name = "yolov8n.pt"

if model_source == "Custom trained model (best.pt)":
    custom_model_path = st.sidebar.text_input(
        "Path to model weights",
        value="models/best.pt",
        help="Path to your custom-trained YOLO best.pt file.",
    )
    if not os.path.exists(custom_model_path):
        st.sidebar.warning(
            f"Model not found at `{custom_model_path}`. "
            "Train one with `python training/train.py --api-key YOUR_KEY`."
        )
        custom_model_path = None
elif model_source == "Roboflow API (local inference)":
    roboflow_api_key = st.sidebar.text_input(
        "Roboflow API Key",
        type="password",
        help="Get a free key at roboflow.com. Needed once to download model; inference is local.",
    )
    if not roboflow_api_key:
        st.sidebar.warning("Enter your Roboflow API key.")
else:
    model_choice = st.sidebar.selectbox(
        "YOLO Model",
        ["yolov8n.pt (fast)", "yolov8s.pt (balanced)", "yolov8m.pt (accurate)"],
        index=0,
    )
    model_name = model_choice.split(" ")[0]

# ---- Detection settings ----
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.3, 0.05)

stride = st.sidebar.slider(
    "Frame stride (speed vs accuracy)",
    min_value=1, max_value=5, value=2, step=1,
    help="Process every Nth frame. 1 = best quality (slow), 2-3 = good balance.",
)

# ---- Ball tracking ----
st.sidebar.header("Ball Tracking")
track_ball = st.sidebar.checkbox("Track the ball", value=True)

# ---- Display options ----
st.sidebar.header("Display Options")
show_trails = st.sidebar.checkbox("Show ball trail", value=True)
trail_length = st.sidebar.slider("Trail length (frames)", 10, 120, 30)

# ---- Speed calibration ----
st.sidebar.header("Speed Calibration")
calibration_mode = st.sidebar.selectbox(
    "Calibration method",
    ["Auto (heuristic)", "Manual (known distance)"],
)
px_per_meter = None
if calibration_mode == "Manual (known distance)":
    st.sidebar.caption("Provide a known real-world distance (e.g. penalty box = 40.3m).")
    known_meters = st.sidebar.number_input("Known distance (meters)", value=40.3)
    known_pixels = st.sidebar.number_input("Pixel distance", value=500)
    if known_pixels > 0:
        px_per_meter = known_pixels / known_meters

# ---- Top-down / combined view ----
st.sidebar.header("Top-Down Pitch View")
enable_topdown = st.sidebar.checkbox(
    "Generate top-down pitch view", value=False,
)

topdown_mode = None
topdown_preset = None
if enable_topdown:
    topdown_mode = st.sidebar.radio(
        "Calibration mode",
        ["Auto (field keypoint detection)", "Manual preset"],
        index=0,
        help=(
            "**Auto**: Uses the Roboflow field-detection model to detect pitch landmarks "
            "and compute perspective transform automatically each frame (best results, "
            "requires API key). "
            "**Manual**: Uses hardcoded presets (approximate)."
        ),
    )
    if topdown_mode == "Manual preset":
        topdown_preset = st.sidebar.selectbox(
            "Camera preset",
            options=get_preset_names(),
            format_func=lambda n: n.replace("_", " ").title(),
        )
    elif topdown_mode == "Auto (field keypoint detection)":
        if not roboflow_api_key:
            td_api_key = st.sidebar.text_input(
                "Roboflow API Key (for field detection)",
                type="password",
                key="td_api_key",
            )
            if td_api_key:
                roboflow_api_key = td_api_key
            else:
                st.sidebar.warning("API key needed for auto field detection.")

enable_combined = False
if enable_topdown:
    enable_combined = st.sidebar.checkbox(
        "Combined video + pitch output", value=True,
        help="Stack the annotated video with the pitch diagram below (like the reference).",
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("⚽ Soccer Match Analyzer")
st.markdown(
    "Upload a short soccer clip to automatically **detect and track players "
    "by team**, track the ball, compute ball speed, and generate a top-down "
    "pitch view. Uses a purpose-built football detection model with jersey "
    "colour clustering for automatic team classification."
)

# ---------------------------------------------------------------------------
# Step 1: Upload video
# ---------------------------------------------------------------------------
st.header("1. Upload Video Clip")

uploaded_file = st.file_uploader(
    "Choose a video file (MP4 recommended, keep it under 30s for fast results)",
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
        for key in ["analysis_done", "tracking_result", "speed_analysis",
                     "annotated_video_path", "topdown_video_path",
                     "combined_video_path", "highlighted_player_id"]:
            st.session_state[key] = None if key != "analysis_done" else False

        progress = st.progress(0, text="Initializing...")
        status_text = st.empty()

        def update_status(msg: str):
            status_text.text(msg)

        try:
            # -- Create tracker --
            progress.progress(10, text="Loading detection model...")
            tracker = SoccerTracker(
                model_path=custom_model_path,
                model_name=model_name,
                confidence=confidence,
            )

            # -- Run tracking --
            progress.progress(20, text="Running detection, tracking & team classification...")
            tracking_result: TrackingResult = tracker.analyze(
                video_path=video_path,
                track_ball=track_ball,
                fps=meta.fps,
                roboflow_api_key=roboflow_api_key if model_source == "Roboflow API (local inference)" else None,
                roboflow_model_id=roboflow_model_id,
                stride=stride,
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

            # -- Extract frames --
            progress.progress(70, text="Extracting frames for annotation...")
            frames = extract_frames(video_path)

            os.makedirs("outputs", exist_ok=True)

            # -- Top-down transformer --
            transformer = None
            if enable_topdown:
                if topdown_mode == "Auto (field keypoint detection)" and roboflow_api_key:
                    progress.progress(72, text="Loading field keypoint model...")
                    try:
                        kp_detector = FieldKeypointDetector(api_key=roboflow_api_key)
                        # Use the first frame to build the transformer
                        if frames:
                            transformer = kp_detector.detect(frames[0])
                            if transformer is None:
                                st.warning("Field keypoints not detected. Falling back to preset.")
                    except Exception as e:
                        st.warning(f"Field detection failed: {e}. Falling back to preset.")

                if transformer is None and topdown_preset:
                    transformer = create_transformer_from_preset(topdown_preset)

            # -- Render output --
            if enable_combined and transformer is not None:
                progress.progress(80, text="Rendering combined video + pitch...")
                combined_frames = annotate_all_frames_with_pitch(
                    frames=frames,
                    tracking_result=tracking_result,
                    transformer=transformer,
                    speed_analysis=speed_analysis,
                    show_trails=show_trails,
                    trail_length=trail_length,
                    progress_callback=update_status,
                )
                combined_path = os.path.join("outputs", "combined_output.mp4")
                frames_to_video(combined_frames, combined_path, meta.fps)
                st.session_state["combined_video_path"] = combined_path
            else:
                # Standard annotated video (no pitch)
                progress.progress(80, text="Annotating frames...")
                annotated_frames = annotate_all_frames(
                    frames=frames,
                    tracking_result=tracking_result,
                    speed_analysis=speed_analysis,
                    highlighted_player_id=None,
                    show_trails=show_trails,
                    trail_length=trail_length,
                    progress_callback=update_status,
                )
                output_path = os.path.join("outputs", "annotated_output.mp4")
                frames_to_video(annotated_frames, output_path, meta.fps)
                st.session_state["annotated_video_path"] = output_path

            # -- Standalone top-down video --
            if enable_topdown and transformer is not None and not enable_combined:
                progress.progress(90, text="Generating top-down pitch view...")
                try:
                    td_frames = render_topdown_video(
                        tracking_result=tracking_result,
                        view_transformer=transformer,
                        progress_callback=update_status,
                    )
                    td_path = os.path.join("outputs", "topdown_output.mp4")
                    frames_to_video(td_frames, td_path, meta.fps)
                    st.session_state["topdown_video_path"] = td_path
                except Exception as td_err:
                    st.warning(f"Top-down rendering failed: {td_err}")

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
if st.session_state.get("analysis_done"):
    st.header("3. Results")

    tracking_result: TrackingResult = st.session_state["tracking_result"]
    speed_analysis: SpeedAnalysis | None = st.session_state["speed_analysis"]

    # -- Summary metrics --
    team_a = tracking_result.get_objects_by_label("team_a")
    team_b = tracking_result.get_objects_by_label("team_b")
    referees = tracking_result.get_objects_by_label("referee")
    ball = tracking_result.get_ball()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Team A Players", len(team_a))
    col2.metric("Team B Players", len(team_b))
    col3.metric("Referees", len(referees))
    col4.metric("Ball Detected", "Yes" if ball else "No")

    # -- Ball possession --
    if tracking_result.ball_possession:
        total_poss = tracking_result.ball_possession.get("team_a", 0) + tracking_result.ball_possession.get("team_b", 0)
        if total_poss > 0:
            pct_a = tracking_result.ball_possession["team_a"] / total_poss * 100
            pct_b = tracking_result.ball_possession["team_b"] / total_poss * 100
            st.subheader("Ball Possession")
            pa, pb = st.columns(2)
            pa.metric("Team A", f"{pct_a:.1f}%")
            pb.metric("Team B", f"{pct_b:.1f}%")

    # -- Combined video + pitch --
    if st.session_state.get("combined_video_path"):
        st.subheader("Combined Video + Pitch View")
        st.caption(
            "Annotated video with a live 2-D pitch diagram below. "
            "Blue = Team A, Red = Team B, Gold = Referees, White = Ball."
        )
        st.video(st.session_state["combined_video_path"])

    # -- Standard annotated video --
    elif st.session_state.get("annotated_video_path"):
        st.subheader("Annotated Video")
        st.video(st.session_state["annotated_video_path"])

    # -- Standalone top-down --
    if st.session_state.get("topdown_video_path"):
        st.subheader("Top-Down Pitch View")
        st.video(st.session_state["topdown_video_path"])

    # -- Ball speed chart --
    if speed_analysis is not None:
        st.subheader("Ball Speed Over Time")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Speed", f"{speed_analysis.avg_speed_kmh} km/h" if speed_analysis.avg_speed_kmh else "N/A")
        col2.metric("Max Speed", f"{speed_analysis.max_speed_kmh} km/h" if speed_analysis.max_speed_kmh else "N/A")
        col3.metric("Min Speed", f"{speed_analysis.min_speed_kmh} km/h" if speed_analysis.min_speed_kmh else "N/A")

        chart_frames = speed_analysis.get_frame_indices()
        chart_speeds = speed_analysis.get_speeds_kmh()
        valid_data = [(f, s) for f, s in zip(chart_frames, chart_speeds) if s is not None]
        if valid_data:
            import pandas as pd
            df = pd.DataFrame(valid_data, columns=["Frame", "Speed (km/h)"])
            df["Time (s)"] = df["Frame"] / tracking_result.fps
            st.line_chart(df, x="Time (s)", y="Speed (km/h)")

    # -- Player list & highlight --
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
                    )
                    highlight_path = os.path.join("outputs", f"highlight_player_{selected_id}.mp4")
                    frames_to_video(annotated_frames, highlight_path, meta.fps)
                    st.video(highlight_path)
        else:
            st.session_state["highlighted_player_id"] = None

    # -- Object table --
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
    "Built with YOLOv8 + ByteTrack + Roboflow Inference | Streamlit | OpenCV | "
    "Columbia DevPost Hackathon 2026"
)
