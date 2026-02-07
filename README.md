# Soccer Match Analyzer ⚽

AI-powered soccer match analysis tool that uses a **purpose-built Roboflow football detection model** (running locally via the `inference` package) for object detection, **ByteTrack** for persistent multi-object tracking, and **K-Means jersey color clustering** for automatic team classification. Falls back to generic YOLOv8 when no API key is provided.

Built for the Columbia DevPost Hackathon 2026.

## Features

-   **Football-Specific Detection** -- Uses the [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) model (ball, goalkeeper, player, referee) running entirely on-device
-   **Team Classification** -- Automatically groups players into two teams based on jersey color using K-Means clustering in HSV color space
-   **Player Tracking** -- Detect and track all players with unique, consistent IDs across frames
-   **Ball Tracking** -- Dedicated ball class from the football model (much better than generic COCO)
-   **Ball Speed Estimation** -- Compute instantaneous ball speed in km/h with optional calibration
-   **Single Player Highlight** -- Select any detected player to highlight with a movement trail
-   **Local Inference** -- Model weights are downloaded once and cached; all processing runs on your machine with no per-frame network calls
-   **Streamlit Dashboard** -- Upload a clip, configure settings, and view annotated results in the browser

## How It Works

```
Upload Video → Detect Players/Ball (local model) → ByteTrack IDs → Jersey Color K-Means → Team A/B → Annotated Video
```

1. You upload a short soccer clip (MP4, ideally under 30 seconds)
2. The Roboflow football model detects players, goalkeepers, the ball, and referees in each frame (locally, no API calls)
3. ByteTrack assigns consistent IDs so each player keeps the same number across frames
4. Jersey colors are sampled from each player's upper body and clustered with K-Means into two teams
5. Ball speed is computed from frame-to-frame centroid displacement
6. Results are rendered as a color-coded annotated video (Team A in orange, Team B in red) with a speed chart

## Prerequisites

-   **Python 3.9 - 3.12** (required by the `inference` package)
-   Works on **CPU**, **Apple Silicon (MPS)**, and **NVIDIA GPUs (CUDA)**
-   A free **Roboflow API key** is needed once to download model weights (get one at [roboflow.com](https://roboflow.com))

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

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Roboflow Setup (Recommended)

The football-specific model provides far better accuracy than generic YOLO for soccer footage. The model weights are downloaded once and cached locally -- all subsequent runs are fully offline.

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Go to your workspace settings and copy your API key
3. In the Streamlit sidebar, check "Use Roboflow football model" (on by default)
4. Paste your API key -- the model downloads once on first run

The model detects four classes:

-   **player** -- field players (distinguished from spectators/staff)
-   **goalkeeper** -- goalkeepers
-   **ball** -- the football
-   **referee** -- match officials (filtered out automatically)

If no API key is provided, the app falls back to generic YOLOv8 (COCO `person` + `sports ball`).

## Project Structure

```
├── app.py                  # Streamlit web application
├── core/
│   ├── __init__.py
│   ├── tracker.py          # YOLOv8/Roboflow + ByteTrack + team classification
│   ├── speed.py            # Ball speed estimation
│   └── visualizer.py       # Frame annotation engine (team-colored boxes)
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
-   **Team colors**: Team A is drawn in orange, Team B in red. The clustering is automatic based on jersey colors
-   **Ball tracking** can be tricky when the ball is very small -- Roboflow models or zoom-in clips work best
-   **Speed calibration**: For accurate km/h readings, use "Manual" calibration and provide a known distance (e.g. penalty box width = 40.3m)

## Performance

| Device              | Model                 | ~FPS (30s clip) |
| ------------------- | --------------------- | --------------- |
| MacBook M1/M2 (MPS) | Roboflow football v11 | ~10-20 FPS      |
| MacBook M1/M2 (CPU) | Roboflow football v11 | ~3-8 FPS        |
| MacBook M1/M2 (MPS) | YOLOv8n (fallback)    | ~15-25 FPS      |
| NVIDIA GPU (CUDA)   | Roboflow football v11 | ~40-60+ FPS     |

All inference runs locally -- no per-frame network calls.

## Tech Stack

| Component           | Technology                                                    |
| ------------------- | ------------------------------------------------------------- |
| Detection           | Roboflow football-players-detection (local) / YOLOv8 fallback |
| Tracking            | ByteTrack (Supervision)                                       |
| Team Classification | K-Means (scikit-learn)                                        |
| Video Processing    | OpenCV                                                        |
| Visualization       | OpenCV + Supervision                                          |
| Speed Computation   | NumPy                                                         |
| Web UI              | Streamlit                                                     |
| Language            | Python 3.9-3.12                                               |

## License

MIT
