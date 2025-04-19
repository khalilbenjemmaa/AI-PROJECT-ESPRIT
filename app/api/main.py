import os
import io
import torch
import gdown
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import base64
import torchvision.transforms as T
import traceback

app = FastAPI()

# CORS settings for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT_PATH = "epoch_30.pt"
GDRIVE_FILE_ID = "1DeTPtUEZI9b1C9f4OsTzGCk-febyUCvW"

# Auto-download from Google Drive
if not os.path.exists(CHECKPOINT_PATH):
    print("⬇️ Downloading model from Google Drive...")
    try:
        gdown.download(id=GDRIVE_FILE_ID, output=CHECKPOINT_PATH, quiet=False)
        print("✅ Download complete.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download checkpoint: {e}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

# Class info
CLASS_NAMES = [
    "Background", "Bottle", "Can", "Chain", "Drink-carton", "Hook",
    "Propeller", "Shampoo-bottle", "Standing-bottle", "Tire", "Valve", "Wall"
]
CLASS_COLORS = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
    (255, 165, 0), (128, 128, 0), (0, 128, 128), (192, 192, 192)
]

# Load model
try:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=1, out_channels=12, init_features=64, pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Image transforms
transform = T.Compose([
    T.Grayscale(),
    T.Resize((256, 256)),
    T.ToTensor()
])

def log_error(e: Exception, file: UploadFile = None):
    error_info = {
        "error": str(e),
        "traceback": traceback.format_exc(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    if file:
        error_info["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
        }
    print("\n🔴 ERROR:")
    for k, v in error_info.items():
        print(f"{k}: {v}")
    return error_info

def create_overlay(original_image, segmentation_mask, alpha=0.5):
    mask_resized = Image.fromarray(segmentation_mask).resize(original_image.size, Image.NEAREST)
    return Image.blend(original_image.convert('RGB'), mask_resized, alpha)

def image_to_base64(img, format="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file.file = io.BytesIO(contents)

        # Validate image
        try:
            img = Image.open(file.file)
            img.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        file.file.seek(0)
        original_image = Image.open(file.file).convert("RGB")
        original_copy = original_image.copy()

        tensor_img = transform(original_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor_img)
            predicted = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        # Prepare result
        present_classes = np.unique(predicted).tolist()
        class_count = {CLASS_NAMES[idx]: np.sum(predicted == idx) for idx in present_classes}
        detected_classes = {name: int(count) for name, count in class_count.items() if name != "Background"}

        color_mask = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
        for idx, color in enumerate(CLASS_COLORS):
            color_mask[predicted == idx] = color

        overlay_image = create_overlay(original_copy, color_mask)

        return JSONResponse({
            "original_image_base64": image_to_base64(original_image),
            "segmentation_map_base64": image_to_base64(Image.fromarray(color_mask)),
            "overlay_image_base64": image_to_base64(overlay_image),
            "classes": detected_classes,
            "class_colors": {CLASS_NAMES[i]: CLASS_COLORS[i] for i in range(len(CLASS_NAMES))}
        })

    except HTTPException:
        raise
    except Exception as e:
        error_info = log_error(e, file)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/class-info/")
async def get_class_info():
    return JSONResponse({
        "class_names": CLASS_NAMES,
        "class_colors": {
            name: {"id": i, "color": color}
            for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
        }
    })

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}
