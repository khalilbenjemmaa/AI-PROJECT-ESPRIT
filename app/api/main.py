import os
import io
import torch
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 Configuration
CHECKPOINT_PATH = "epoch_30.pt"
GDRIVE_FILE_ID = "1DeTPtUEZI9b1C9f4OsTzGCk-febyUCvW"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# ⬇️ Auto-download model checkpoint
if not os.path.exists(CHECKPOINT_PATH):
    print("⬇️ Downloading model from Google Drive...")
    response = requests.get(GDRIVE_URL)
    if response.status_code == 200:
        with open(CHECKPOINT_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded.")
    else:
        raise RuntimeError(f"❌ Failed to download model: HTTP {response.status_code}")

# 🔤 Class info
CLASS_NAMES = [
    "Background", "Bottle", "Can", "Chain", "Drink-carton", "Hook",
    "Propeller", "Shampoo-bottle", "Standing-bottle", "Tire", "Valve", "Wall"
]

CLASS_COLORS = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
    (255, 165, 0), (128, 128, 0), (0, 128, 128), (192, 192, 192)
]

# ⚙️ Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

# 🧠 Load model
try:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=1, out_channels=12, init_features=64, pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# 📸 Image preprocessing
transform = T.Compose([
    T.Grayscale(),
    T.Resize((256, 256)),
    T.ToTensor()
])

def log_error(e: Exception, file: UploadFile = None):
    return {
        "error": str(e),
        "traceback": traceback.format_exc(),
        "system_info": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device)
        },
        "file_info": {
            "filename": file.filename if file else None,
            "content_type": file.content_type if file else None
        }
    }

def create_overlay(original_image, segmentation_mask, alpha=0.5):
    mask_resized = Image.fromarray(segmentation_mask).resize(original_image.size, Image.NEAREST)
    return Image.blend(original_image.convert('RGB'), mask_resized, alpha)

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file.file = io.BytesIO(contents)
        try:
            img = Image.open(file.file)
            img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        file.file.seek(0)
        original = Image.open(file.file).convert("RGB")
        tensor_img = transform(original).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            predicted = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        present_classes = np.unique(predicted).tolist()
        class_count = {CLASS_NAMES[i]: np.sum(predicted == i) for i in present_classes}
        detected = {k: int(v) for k, v in class_count.items() if k != "Background"}

        # Color segmentation mask
        mask = np.zeros((256, 256, 3), dtype=np.uint8)
        for i, color in enumerate(CLASS_COLORS):
            mask[predicted == i] = color

        overlay = create_overlay(original, mask)

        def img_to_base64(img, fmt="PNG"):
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            return base64.b64encode(buf.getvalue()).decode()

        return JSONResponse({
            "original_image_base64": img_to_base64(original),
            "segmentation_map_base64": img_to_base64(Image.fromarray(mask)),
            "overlay_image_base64": img_to_base64(overlay),
            "classes": detected,
            "class_colors": {CLASS_NAMES[i]: CLASS_COLORS[i] for i in range(len(CLASS_NAMES))}
        })

    except HTTPException:
        raise
    except Exception as e:
        error = log_error(e, file)
        print("❌ Error:", error)
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/class-info/")
async def get_class_info():
    return {
        "class_names": CLASS_NAMES,
        "class_colors": {name: {"id": idx, "color": color} for idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))}
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}
