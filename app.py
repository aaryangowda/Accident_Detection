from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import tensorflow as tf

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from detection import AccidentDetectionModel
from dotenv import load_dotenv
import io
import uvicorn
import gdown
import logging
import shutil
from typing import Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Model file paths
MODEL_JSON = "model.json"
MODEL_WEIGHTS = "model_weights.h5"

# Google Drive file IDs
MODEL_JSON_ID = os.getenv("MODEL_JSON_ID")
MODEL_WEIGHTS_ID = os.getenv("MODEL_WEIGHTS_ID")

# Video file path - using a relative path
VIDEO_PATH = "static/video.mp4"
UPLOAD_DIR = "static/uploads"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def download_model_files():
    """Download model files from Google Drive if they don't exist"""
    try:
        # Construct direct Google Drive URLs
        json_url = f"https://drive.google.com/uc?id={MODEL_JSON_ID}"
        weights_url = f"https://drive.google.com/uc?id={MODEL_WEIGHTS_ID}"
        
        if not MODEL_JSON_ID or not MODEL_WEIGHTS_ID:
            raise ValueError("MODEL_JSON_ID and MODEL_WEIGHTS_ID must be set in .env file")
            
        if not os.path.exists(MODEL_JSON):
            logger.info("Downloading model.json...")
            gdown.download(url=json_url, output=MODEL_JSON, quiet=False)
            if not os.path.exists(MODEL_JSON):
                raise Exception("Failed to download model.json")
        
        if not os.path.exists(MODEL_WEIGHTS):
            logger.info("Downloading model_weights.h5...")
            gdown.download(url=weights_url, output=MODEL_WEIGHTS, quiet=False)
            if not os.path.exists(MODEL_WEIGHTS):
                raise Exception("Failed to download model_weights.h5")
                
        logger.info("Model files downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        raise

# Download model files on startup
download_model_files()

# Initialize model
try:
    model = AccidentDetectionModel(MODEL_JSON, MODEL_WEIGHTS)
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

font = cv2.FONT_HERSHEY_SIMPLEX

# Global variables for video capture
video_capture = None
current_video_path = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global video_capture, current_video_path
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return JSONResponse(
                status_code=400,
                content={"error": "Only MP4, AVI, and MOV files are allowed"}
            )
        
        # Create a unique filename
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update current video path
        current_video_path = file_path
        
        # Release existing video capture if any
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        
        logger.info(f"Video uploaded successfully: {file_path}")
        return JSONResponse(content={"message": "Video uploaded successfully"})
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error uploading video: {str(e)}"}
        )

@app.get("/video_feed")
async def video_feed():
    global video_capture, current_video_path
    
    try:
        if video_capture is None:
            logger.info("Initializing video capture...")
            
            # Try to use uploaded video if available
            if current_video_path and os.path.exists(current_video_path):
                video_source = current_video_path
            else:
                # Try webcam first
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    logger.info("Webcam not available, trying default video file...")
                    video_source = VIDEO_PATH
                else:
                    video_source = 0
            
            video_capture = cv2.VideoCapture(video_source)
            if not video_capture.isOpened():
                error_msg = "Could not open any video source"
                logger.error(error_msg)
                return Response(content=error_msg, status_code=500)
            logger.info("Video capture initialized successfully")
        
        def generate_frames():
            while True:
                try:
                    ret, frame = video_capture.read()
                    if not ret:
                        logger.info("End of video reached, restarting...")
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                        
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    roi = cv2.resize(gray_frame, (250, 250))
                    
                    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
                    if pred == "Accident":
                        prob = (round(prob[0][0]*100, 2))
                        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                        cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)
                    
                    # Convert frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logger.error(f"Error generating frame: {str(e)}")
                    break
        
        return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        error_msg = f"Error in video feed: {str(e)}"
        logger.error(error_msg)
        return Response(content=error_msg, status_code=500)

@app.on_event("shutdown")
async def shutdown_event():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        logger.info("Video capture released")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True) 
