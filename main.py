import os
import uuid
import shutil
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import MesoNetDetector

# --- 1. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepfakeAPI")

# --- 2. ROBUST PATH HANDLING ---
# Get the absolute path to the current folder (where main.py lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute path: /opt/render/project/src/models/mesonet_whisper_mfcc_finetuned.pth
MODEL_PATH = os.path.join(BASE_DIR, "models", "mesonet_whisper_mfcc_finetuned.pth")

# --- 3. LIFESPAN MANAGEMENT ---
# This ensures the model loads once when the server starts and cleans up on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ CRITICAL: Model file not found at {MODEL_PATH}")
        # We don't raise an error here so the server can at least start and show logs
        app.state.detector = None
    else:
        logger.info(f"✅ SUCCESS: Loading model from {MODEL_PATH}")
        app.state.detector = MesoNetDetector(model_path=MODEL_PATH)
    
    yield
    # Shutdown Logic (Cleanup)
    if hasattr(app.state, 'detector'):
        del app.state.detector

# --- 4. INITIALIZE APP ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your Lovable domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. ENDPOINTS ---

@app.get("/")
async def health_check():
    status = "Online" if app.state.detector else "Error: Model Missing"
    return {
        "status": status,
        "model_path_checked": MODEL_PATH,
        "file_exists": os.path.exists(MODEL_PATH)
    }

@app.post("/detect")
async def detect_audio(file: UploadFile = File(...)):
    if not app.state.detector:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    # Create a unique temp file
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(BASE_DIR, f"temp_{file_id}_{file.filename}")
    
    try:
        # Save uploaded bytes to temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run prediction using the detector stored in app state
        results = app.state.detector.predict(temp_path)
        
        return results

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    finally:
        # Always delete the temp file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)