from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from inference import MesoNetDetector

app = FastAPI()

# CRITICAL: Allow your Lovable site to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your actual Lovable domain later for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your detector (ensure the path to weights is correct)
detector = MesoNetDetector(model_path="models/mesonet_whisper_mfcc_finetuned.pth")

@app.post("/detect")
async def detect_audio(file: UploadFile = File(...)):
    # 1. Create a unique filename to handle multiple users
    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}_{file.filename}"
    
    try:
        # 2. Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Run your prediction
        results = detector.predict(temp_path)
        
        # 4. Cleanup
        os.remove(temp_path)
        
        return results
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "Detector is online"}