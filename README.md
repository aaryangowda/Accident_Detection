# Accident Detection System

A real-time accident detection system using deep learning and computer vision. The system can process video feeds from webcams or uploaded video files to detect accidents in real-time.

## Features

- Real-time accident detection
- Support for webcam input
- Video file upload functionality
- Web-based interface
- Real-time probability display
- Debug information
- Supports MP4, AVI, and MOV video formats

## Tech Stack

- Python 3.12
- FastAPI
- OpenCV
- TensorFlow/Keras
- HTML/CSS/JavaScript
- Bootstrap 5

## Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd accident-detection-system
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Files

The system requires two model files:
- `model.json`: Model architecture
- `model_weights.h5`: Model weights

These files will be automatically downloaded from Google Drive when you run the application.

## Running Locally

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:8080
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Add the following environment variables:
     - `PYTHON_VERSION`: 3.12

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```env
MODEL_JSON_ID=your_model_json_id
MODEL_WEIGHTS_ID=your_model_weights_id
```

## Project Structure

```
.
├── app.py              # Main FastAPI application
├── detection.py        # Accident detection model implementation
├── requirements.txt    # Python dependencies
├── static/            # Static files
│   └── uploads/       # Directory for uploaded videos
├── templates/         # HTML templates
│   └── index.html    # Main web interface
└── README.md         # This file
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
