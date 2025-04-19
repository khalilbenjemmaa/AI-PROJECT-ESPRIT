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
import gdown

app = FastAPI()

# Add CORS to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT_PATH = "epoch_30.pt"

# 🔁 Replace this with your actual Google Drive file ID
GDRIVE_FILE_ID = "1DeTPtUEZI9b1C9f4OsTzGCk-febyUCvW"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# Auto-download model if not found
if not os.path.exists(CHECKPOINT_PATH):
    print("⬇️ Downloading model checkpoint from Google Drive...")
    response = gdown.download(id=GDRIVE_FILE_ID, output=CHECKPOINT_PATH, quiet=False)
    if response:
        print("✅ Model checkpoint downloaded successfully.")
    else:
        raise RuntimeError(f"Failed to download checkpoint from Google Drive.")

CLASS_NAMES = [
    "Background", "Bottle", "Can", "Chain", "Drink-carton", "Hook",
    "Propeller", "Shampoo-bottle", "Standing-bottle", "Tire", "Valve", "Wall"
]

CLASS_COLORS = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
    (255, 165, 0), (128, 128, 0), (0, 128, 128), (192, 192, 192)
]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n⚙️ Using device: {device}\n")

# Model loading with validation
try:
    # Load the model architecture
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                          in_channels=1, out_channels=12, init_features=64, pretrained=False)

    # Use `weights_only=False` to load the entire model (architecture + weights)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    # Load state dict for model
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval().to(device)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

# Image transforms
transform = T.Compose([
    T.Grayscale(),
    T.Resize((256, 256)),
    T.ToTensor()
])

def log_error(e: Exception, file: UploadFile = None):
    error_details = {
        "error": str(e),
        "traceback": traceback.format_exc(),
        "system_info": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device)
        }
    }
    
    if file:
        error_details["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size
        }
    
    print("\n🔴 ERROR DETAILS:")
    for k, v in error_details.items():
        print(f"│ {k.upper():<15}: {v}")
    
    return error_details

def create_overlay(original_image, segmentation_mask, alpha=0.5):
    mask_resized = Image.fromarray(segmentation_mask).resize(original_image.size, Image.NEAREST)
    overlay = Image.blend(original_image.convert('RGB'), mask_resized, alpha)
    return overlay

@app.post("/predict/")  # POST method to make predictions
async def predict_segmentation(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file.file = io.BytesIO(contents)
        
        try:
            original_image = Image.open(file.file)
            original_image.verify()  # Verifies the image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        file.file.seek(0)
        original_image = Image.open(file.file).convert("RGB")
        original_image_copy = original_image.copy()

        tensor_img = transform(original_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            predicted = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        present_classes = np.unique(predicted).tolist()
        class_count = {CLASS_NAMES[idx]: np.sum(predicted == idx) for idx in present_classes}
        detected_classes = {CLASS_NAMES[idx]: int(class_count[CLASS_NAMES[idx]]) 
                          for idx in present_classes if idx != 0}

        color_mask = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
        for class_index, color in enumerate(CLASS_COLORS):
            color_mask[predicted == class_index] = color

        overlay_image = create_overlay(original_image_copy, color_mask)
        
        def image_to_base64(img, format="PNG"):
            buffered = io.BytesIO()
            img.save(buffered, format=format)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
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

@app.get("/class-info/")  # GET method to retrieve class info
async def get_class_info():
    color_dict = {
        name: {"color": color, "id": i}
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
    }
    
    return JSONResponse({
        "class_names": CLASS_NAMES,
        "class_colors": color_dict
    })

@app.get("/health")  # Health check endpoint
async def health_check():
    return {"status": "ok", "model_loaded": True}
