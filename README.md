# Soccer Match Analyzer ⚽

AI-powered soccer match analysis tool that uses **Meta SAM 3** (Segment Anything Model 3) to track players, the ball, and compute ball speed from short video clips.

Built for the Columbia DevPost Hackathon 2026.

## Features

-   **Player Tracking** -- Automatically detect and track all players from both teams using text prompts (e.g. "player in white jersey")
-   **Ball Tracking** -- Track the soccer ball across every frame
-   **Ball Speed Estimation** -- Compute instantaneous ball speed in km/h with optional calibration
-   **Single Player Highlight** -- Select any detected player to highlight with a movement trail
-   **Streamlit Dashboard** -- Upload a clip, configure prompts, and view annotated results in the browser

## How It Works

```
Upload Video → SAM 3 Text Prompts → Per-frame Tracking → Speed Calculation → Annotated Video + Charts
```

1. You upload a short soccer clip (MP4, ideally under 30 seconds)
2. SAM 3's video predictor receives text prompts like "player in white jersey", "player in red jersey", and "soccer ball"
3. SAM 3 detects all matching objects in the first frame and tracks them across every subsequent frame with consistent IDs
4. Ball speed is computed from frame-to-frame centroid displacement
5. Results are rendered as an annotated video with bounding boxes, trails, and a speed chart

## Prerequisites

-   **Python 3.12+**
-   **PyTorch 2.7+** with CUDA 12.6+
-   **CUDA-compatible GPU** (required for SAM 3 inference)
-   **HuggingFace account** with access to SAM 3 weights

## Setup

### 1. Create conda environment

```bash
conda create -n soccer-analyzer python=3.12
conda activate soccer-analyzer
```

### 2. Install PyTorch with CUDA

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Authenticate with HuggingFace

Request access to SAM 3 model weights at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then:

```bash
pip install huggingface-hub
huggingface-cli login
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
├── app.py                  # Streamlit web application
├── core/
│   ├── __init__.py
│   ├── tracker.py          # SAM 3 video tracking wrapper
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
-   **Be specific with team prompts** -- "player in white jersey with blue shorts" works better than just "player"
-   **Ball tracking** can be tricky when the ball is very small -- zoom-in clips work best
-   **Speed calibration**: For accurate km/h readings, use "Manual" calibration and provide a known distance (e.g. penalty box width = 40.3m)

## Tech Stack

| Component         | Technology                            |
| ----------------- | ------------------------------------- |
| CV/ML             | Meta SAM 3 (Segment Anything Model 3) |
| Video Processing  | OpenCV                                |
| Visualization     | OpenCV + Supervision                  |
| Speed Computation | NumPy                                 |
| Web UI            | Streamlit                             |
| Language          | Python 3.12                           |

## No GPU? Alternatives

-   **Google Colab**: Run the analysis in Colab (free T4 GPU) and download the annotated video
-   **Cloud VM**: Spin up a GPU instance on AWS/GCP/Azure

## License

MIT
