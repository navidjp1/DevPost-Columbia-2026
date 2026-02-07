# Soccer Match Analyzer ⚽

AI-powered soccer match analysis tool that uses **YOLOv8** for object detection and **ByteTrack** for persistent multi-object tracking to analyze soccer video clips.

Built for the Columbia DevPost Hackathon 2026.

## Features

-   **Player Tracking** -- Automatically detect and track all players with unique, consistent IDs across frames
-   **Ball Tracking** -- Track the soccer ball across every frame
-   **Ball Speed Estimation** -- Compute instantaneous ball speed in km/h with optional calibration
-   **Single Player Highlight** -- Select any detected player to highlight with a movement trail
-   **Streamlit Dashboard** -- Upload a clip, configure settings, and view annotated results in the browser

## How It Works

```
Upload Video → YOLOv8 Detection → ByteTrack ID Assignment → Speed Calculation → Annotated Video + Charts
```

1. You upload a short soccer clip (MP4, ideally under 30 seconds)
2. YOLOv8 detects all people and sports balls in each frame
3. ByteTrack assigns consistent IDs so each player keeps the same number across frames
4. Ball speed is computed from frame-to-frame centroid displacement
5. Results are rendered as an annotated video with bounding boxes, trails, and a speed chart

## Prerequisites

-   **Python 3.9+**
-   Works on **CPU**, **Apple Silicon (MPS)**, and **NVIDIA GPUs (CUDA)**
-   No special authentication or model access required -- YOLO weights download automatically

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# or: venv\Scripts\activate  # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

That's it! YOLOv8 model weights (~6MB for nano) download automatically on first run.

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
├── app.py                  # Streamlit web application
├── core/
│   ├── __init__.py
│   ├── tracker.py          # YOLOv8 + ByteTrack tracking engine
│   ├── speed.py            # Ball speed estimation
│   └── visualizer.py       # Frame annotation engine
├── utils/
│   ├── __init__.py
│   └── video.py            # Video I/O helpers
├── outputs/                # Generated annotated videos (gitignored)
├── requirements.txt        # Python dependencies
└── README.md
```

## Usage Tips

-   **Keep clips short** (10-30 seconds) for fast processing during the hackathon demo
-   **YOLO model selection**: Use `yolov8n` (nano) for speed, `yolov8s` (small) for a balance, or `yolov8m` (medium) for best accuracy
-   **Confidence threshold**: Lower it (e.g. 0.2) if players are being missed; raise it (e.g. 0.5) if non-players are being detected
-   **Ball tracking** can be tricky when the ball is very small -- zoom-in clips work best
-   **Speed calibration**: For accurate km/h readings, use "Manual" calibration and provide a known distance (e.g. penalty box width = 40.3m)

## Performance

| Device              | YOLO Model | ~FPS (30s clip) |
| ------------------- | ---------- | --------------- |
| MacBook M1/M2 (MPS) | yolov8n    | ~15-25 FPS      |
| MacBook M1/M2 (CPU) | yolov8n    | ~5-10 FPS       |
| NVIDIA GPU (CUDA)   | yolov8n    | ~60+ FPS        |
| NVIDIA GPU (CUDA)   | yolov8m    | ~30-40 FPS      |

## Tech Stack

| Component         | Technology              |
| ----------------- | ----------------------- |
| Detection         | YOLOv8 (Ultralytics)    |
| Tracking          | ByteTrack (Supervision) |
| Video Processing  | OpenCV                  |
| Visualization     | OpenCV + Supervision    |
| Speed Computation | NumPy                   |
| Web UI            | Streamlit               |
| Language          | Python 3.9+             |

## License

MIT
