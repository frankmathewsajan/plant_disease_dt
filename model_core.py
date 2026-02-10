import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------
# MODEL REGISTRY (loaded on demand)
# -------------------------------------------------

MODEL_DIR = "./models"

SUPPORTED_CROPS = {
    "chilli",
    "cotton",
    "rose",
    "tomato",
    "turmeric",
}

_loaded_models = {}


def get_model(crop: str) -> YOLO:
    """
    Loads and caches YOLO models.
    """
    crop = crop.lower()

    if crop not in SUPPORTED_CROPS:
        raise ValueError(f"Unsupported crop: {crop}")

    if crop not in _loaded_models:
        model_path = f"{MODEL_DIR}/{crop}.pt"
        _loaded_models[crop] = YOLO(model_path)

    return _loaded_models[crop]


# -------------------------------------------------
# IMAGE LOADER
# -------------------------------------------------

def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid image path or corrupted image")
    return img


# -------------------------------------------------
# INFERENCE ENGINE (DETECTION + CLASSIFICATION)
# -------------------------------------------------

def run_inference(
    crop: str,
    image,
    threshold: float = 0.5,
):
    """
    Returns unified inference result.

    Output format:
    {
      "crop": "tomato",
      "boxes": [...],
      "classification": [...]
    }
    """

    model = get_model(crop)
    result = model(image, verbose=False)[0]

    output = {
        "crop": crop,
        "boxes": [],
        "classification": [],
    }

    # ----------- DETECTION MODELS -----------
    if hasattr(result, "boxes") and result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                continue

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            output["boxes"].append({
                "class": result.names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

    # ----------- CLASSIFICATION MODELS -----------
    if hasattr(result, "probs") and result.probs is not None:
        probs = result.probs.data.tolist()
        for i, p in enumerate(probs):
            if p < threshold:
                continue

            output["classification"].append({
                "class": result.names[i],
                "confidence": float(p),
            })

    return output


# -------------------------------------------------
# OPTIONAL: ANNOTATED IMAGE (FOR UI)
# -------------------------------------------------

def run_inference_with_plot(
    crop: str,
    image,
    threshold: float = 0.5,
):
    model = get_model(crop)
    result = model(image, verbose=False)[0]
    annotated_image = result.plot()

    data = run_inference(crop, image, threshold)
    return annotated_image, data


# -------------------------------------------------
# TEST (SAFE TO DELETE)
# -------------------------------------------------

if __name__ == "__main__":
    img = load_image("test.jpg")
    out = run_inference("tomato", img, 0.4)
    print(out)
