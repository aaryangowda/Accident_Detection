from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import tensorflow as tf

# Force TensorFlow to use CPU and optimize memory usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Configure TensorFlow for minimal memory usage
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from detection import AccidentDetectionModel
from dotenv import load_dotenv
import io
import uvicorn
import gdown
import logging
import shutil
import gc
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
MODEL_JSON_ID = os.getenv("MODEL_JSON_ID", "1rTNqBBjEE9XnuWM8FFInOI_1o3Skw7xa")
MODEL_WEIGHTS_ID = os.getenv("MODEL_WEIGHTS_ID", "18dLdwQiubd0yqnNpkM5Pi6G6EM0PxXKg")

# Video file path
VIDEO_PATH = "static/video.mp4"
UPLOAD_DIR = "static/uploads"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables
model = None
font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = None
current_video_path = None
frame_skip = 4  # Process every 4th frame
frame_count = 0
MAX_FRAME_SIZE = (120, 90)  # Reduced frame size
JPEG_QUALITY = 30  # Reduced JPEG quality

def initialize_model():
    """Initialize the model with memory optimization"""
    global model
    try:
        if model is None:
            model = AccidentDetectionModel(MODEL_JSON, MODEL_WEIGHTS)
            logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def download_model_files():
    """Download model files from Google Drive if they don't exist"""
    try:
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
        
        # Initialize model after downloading
        initialize_model()
        
    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        raise

# Download model files on startup
download_model_files()

def process_frame(frame):
    """Process a single frame with memory optimization"""
    try:
        # Reduce frame size immediately
        frame = cv2.resize(frame, MAX_FRAME_SIZE)
        
        # Convert to RGB and prepare ROI (maintain aspect ratio)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        
        # Free memory
        del rgb_frame
        
        # Normalize
        roi = (roi.astype(np.float32) / 255.0)
        
        # Make prediction
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        
        # Free memory
        del roi
        
        # Draw prediction if accident detected
        if pred == "Accident":
            prob = round(prob[0][0]*100, 2)
            cv2.rectangle(frame, (0, 0), (60, 12), (0, 0, 0), -1)
            cv2.putText(frame, f"{prob}%", (2, 10), font, 0.3, (255, 255, 0), 1)
        
        # Convert to JPEG with very low quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()
        
        # Clean up
        del frame
        del buffer
        gc.collect()
        
        return frame_bytes
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

def generate_frames():
    global frame_count
    while True:
        try:
            # Clear memory periodically
            if frame_count % (frame_skip * 5) == 0:
                tf.keras.backend.clear_session()
                gc.collect()
            
            ret, frame = video_capture.read()
            if not ret:
                logger.info("End of video reached, restarting...")
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            
            # Skip frames to reduce processing load
            frame_count += 1
            if frame_count % frame_skip != 0:
                del frame
                continue
            
            frame_bytes = process_frame(frame)
            del frame
            
            if frame_bytes is None:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            del frame_bytes
            
        except Exception as e:
            logger.error(f"Error generating frame: {str(e)}")
            break

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
