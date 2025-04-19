from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import uvicorn

app = FastAPI()

# Allow CORS from any origin (helpful for Wix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8-tiny model
yolo_model = YOLO("yolov8n.pt")

class ImagePayload(BaseModel):
    image: str  # base64 encoded JPEG

def detect_clutter_yolo(image_np):
    results = yolo_model(image_np)
    objects = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = yolo_model.model.names[cls_id]
            objects.append(label)
    unique_objects = set(objects)
    clutter_level = "low"
    if len(objects) >= 5:
        clutter_level = "high"
    elif len(objects) >= 2:
        clutter_level = "moderate"
    return clutter_level, list(unique_objects)

@app.post("/analyze-frame")
async def analyze_frame(payload: ImagePayload):
    try:
        image_data = base64.b64decode(payload.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        clutter_level, objects = detect_clutter_yolo(image_np)

        return {
            "clutter": clutter_level,
            "objects": objects,
            "summary": f"Detected {len(objects)} object(s)"
        }
    except Exception as e:
        return {"error": str(e)}

# Run locally with: python server_backend.py
if __name__ == "__main__":
    uvicorn.run("server_backend:app", host="0.0.0.0", port=8000, reload=True)