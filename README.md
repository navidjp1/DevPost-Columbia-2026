# ⚽ Soccer Match Analyzer

AI-powered soccer match analysis tool that detects and tracks players, assigns
teams automatically, tracks the ball, computes ball speed, and generates a
top-down pitch view -- all from a short video clip.

**Based on**: [zakroum-hicham/football-analysis-CV](https://github.com/zakroum-hicham/football-analysis-CV)

## Features

-   **Player detection & tracking** -- YOLOv8 + ByteTrack (supervision)
-   **Automatic team classification** -- KMeans clustering on jersey colours (HSV)
-   **Ball tracking** -- with interpolation to fill gaps during passes/shots
-   **Ball possession** -- real-time ball-to-player distance assignment
-   **Ball speed estimation** -- with optional pixel-to-meter calibration
-   **Referee tracking** -- detects and tracks all referees
-   **Top-down pitch view** -- automatic perspective transform using Roboflow
    field keypoint detection model, or manual presets
-   **Combined output** -- annotated video stacked with a live 2-D pitch diagram
-   **Streamlit dashboard** -- upload, configure, and view results in the browser

## How It Works

1. **Detection**: A custom-trained YOLOv8 model (or Roboflow football model)
   detects players, goalkeepers, the ball, and referees.
2. **Tracking**: Supervision ByteTrack assigns persistent IDs across frames.
3. **Team Assignment**: Top-half jersey crops are clustered with KMeans (HSV)
   to separate background from jersey colour, then all players are clustered
   into 2 teams.
4. **Ball Possession**: The closest player to the ball (within 70px) is assigned
   possession each frame. Accumulated per-team.
5. **Top-Down View**: The Roboflow `football-field-detection-f07vi/14` model
   detects 32 pitch keypoints. A homography maps pixel coords to real-world
   pitch coords (cm). Players and ball are drawn on a 2-D pitch diagram.
6. **Annotation**: Supervision ellipse, triangle, and label annotators produce
   clean broadcast-style overlays.

## Prerequisites

-   Python 3.10+
-   A Roboflow API key (free at [roboflow.com](https://roboflow.com))
-   (Optional) A GPU for faster inference and model training

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd DevPostColumbia2026

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Training Your Own Model

For best results, train a YOLOv8 model on the Roboflow football-players-detection
dataset. This gives you a `best.pt` model that detects ball, goalkeeper, player,
and referee natively.

```bash
# Train (adjust --model, --epochs, --imgsz for your hardware)
python training/train.py --api-key YOUR_ROBOFLOW_KEY

# For GPU (recommended):
python training/train.py --api-key YOUR_KEY --model yolov8x.pt --epochs 100 --imgsz 1280

# For CPU (slower but works):
python training/train.py --api-key YOUR_KEY --model yolov8s.pt --epochs 50 --imgsz 640
```

The trained weights are saved to `models/best.pt` and can be selected in the
Streamlit app.

## Project Structure

```
├── app.py                  # Streamlit web application
├── core/
│   ├── tracker.py          # Detection, tracking, team assignment, ball possession
│   ├── visualizer.py       # Frame annotation with supervision annotators
│   ├── view_transformer.py # Perspective transform & pitch drawing
│   └── speed.py            # Ball speed estimation
├── training/
│   └── train.py            # YOLO fine-tuning script
├── models/                 # Trained model weights (best.pt)
├── utils/
│   └── video.py            # Video I/O utilities
├── outputs/                # Generated videos
└── requirements.txt
```

## Tech Stack

| Component             | Library                                                           |
| --------------------- | ----------------------------------------------------------------- |
| Object Detection      | YOLOv8 (ultralytics) / Roboflow inference                         |
| Object Tracking       | ByteTrack (supervision)                                           |
| Team Classification   | KMeans (scikit-learn)                                             |
| Field Detection       | Roboflow football-field-detection-f07vi/14                        |
| Annotations           | supervision (EllipseAnnotator, TriangleAnnotator, LabelAnnotator) |
| Perspective Transform | OpenCV (findHomography / getPerspectiveTransform)                 |
| Web UI                | Streamlit                                                         |
| Ball Interpolation    | pandas                                                            |

## Credits

-   [zakroum-hicham/football-analysis-CV](https://github.com/zakroum-hicham/football-analysis-CV) -- reference implementation
-   [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis) -- ball interpolation approach
-   [Roboflow](https://roboflow.com) -- football-players-detection and field-detection models
-   [supervision](https://github.com/roboflow/supervision) -- tracking and annotation toolkit
