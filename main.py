from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import numpy as np
import cv2
import sys
import torch
import ultralytics

from model_core import run_inference, SUPPORTED_CROPS, _loaded_models

app = FastAPI(
    title="Crop Disease Detection API",
    version="1.0.0",
    description="""
YOLO-based inference API for crop disease detection.

### Supported Crops
- chilli
- cotton
- rose
- tomato
- turmeric

### Output Types
- **boxes** → Bounding box detection
- **classify** → Classification probabilities
"""
)

# --------------------------------------------------
# ROOT INDEX
# --------------------------------------------------

@app.get(
    "/",
    summary="API index",
    include_in_schema=False
)
def root():
    """
    Redirects to interactive API documentation.
    """
    return {
        "message": "Crop Disease Detection API is running",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "predict": "/predict/{crop}",
            "health": "/health",
        }
    }

# --------------------------------------------------
# UTILS
# --------------------------------------------------

def read_image_from_upload(file: UploadFile):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Unsupported image format")

    data = np.frombuffer(file.file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image")

    return image


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.post(
    "/predict/{crop}",
    summary="Crop disease inference",
)
async def predict(
    crop: str,
    file: UploadFile = File(..., description="Input image"),
    output_type: str = Query(
        "boxes",
        description="Inference type: boxes | classify",
    ),
    threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold (0–1)",
    ),
):
    """
    ### Request Parameters
    - **crop**: Crop name
    - **file**: Image file (jpg / png)
    - **output_type**:
        - `boxes` → object detection
        - `classify` → classification
    - **threshold**: Confidence threshold

    ### Response
    JSON result with detection or classification output.
    """

    crop = crop.lower()
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported crop. Choose from {sorted(SUPPORTED_CROPS)}",
        )

    image = read_image_from_upload(file)
    result = run_inference(crop, image, threshold)

    return JSONResponse({
        "crop": crop,
        "boxes": result["boxes"],
        "classification": result["classification"] if output_type == "classify" else []
    })


# --------------------------------------------------
# HEALTH CHECK (DETAILED)
# --------------------------------------------------

@app.get(
    "/health",
    summary="Detailed health check",
)
def health():
    return {
        "status": "ok",
        "api_version": app.version,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "supported_crops": sorted(SUPPORTED_CROPS),
        "models_loaded_in_cache": sorted(list(_loaded_models.keys())),
        "model_cache_size": len(_loaded_models),
    }
