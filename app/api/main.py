import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import base64
import torchvision.transforms as T
import traceback

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
CHECKPOINT_PATH = r"C:\Users\HKrid\OneDrive - Linedata Services, Inc\Desktop\seg\trained_modal\epoch_30.pt"
VALID_IMAGE_PATH = r"C:\Users\HKrid\OneDrive - Linedata Services, Inc\Desktop\seg\data\Images\marine-debris-aris3k-1.png"

CLASS_NAMES = [
    "Background", "Bottle", "Can", "Chain", "Drink-carton", "Hook",
    "Propeller", "Shampoo-bottle", "Standing-bottle", "Tire", "Valve", "Wall"
]

CLASS_COLORS = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
    (255, 165, 0), (128, 128, 0), (0, 128, 128), (192, 192, 192)
]

# Path validation with meaningful error messages
if not os.path.exists(CHECKPOINT_PATH):
    raise ValueError(f"Checkpoint not found at {CHECKPOINT_PATH}")
if not os.path.exists(VALID_IMAGE_PATH):
    raise ValueError(f"Sample image not found at {VALID_IMAGE_PATH}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n⚙️ Using device: {device}\n")

# Model loading with validation
try:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                          in_channels=1, out_channels=12, init_features=64, pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
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
    """Log detailed error information"""
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
    """Create an overlay of segmentation on the original image with transparency"""
    # Resize mask to match original image dimensions
    mask_resized = Image.fromarray(segmentation_mask).resize(original_image.size, Image.NEAREST)
    
    # Create overlay
    overlay = Image.blend(original_image.convert('RGB'), mask_resized, alpha)
    
    return overlay

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    try:
        # Load and verify image
        contents = await file.read()
        file.file = io.BytesIO(contents)  # Reset file pointer after reading
        
        try:
            original_image = Image.open(file.file)
            original_image.verify()  # Verify image integrity
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Reset file pointer and open again
        file.file.seek(0)
        original_image = Image.open(file.file).convert("RGB")
        original_image_copy = original_image.copy()  # Make a copy for overlay

        # Apply transforms for model input
        tensor_img = transform(original_image).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            output = model(tensor_img)
            predicted = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        # Process results
        present_classes = np.unique(predicted).tolist()
        class_count = {CLASS_NAMES[idx]: np.sum(predicted == idx) for idx in present_classes}
        detected_classes = {CLASS_NAMES[idx]: int(class_count[CLASS_NAMES[idx]]) 
                          for idx in present_classes if idx != 0}

        # Generate color mask
        color_mask = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
        for class_index, color in enumerate(CLASS_COLORS):
            color_mask[predicted == class_index] = color

        # Create overlay image
        overlay_image = create_overlay(original_image_copy, color_mask)
        
        # Convert all images to base64
        def image_to_base64(img, format="PNG"):
            buffered = io.BytesIO()
            img.save(buffered, format=format)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Return all needed images and data
        return JSONResponse({
            "original_image_base64": image_to_base64(original_image),
            "segmentation_map_base64": image_to_base64(Image.fromarray(color_mask)),
            "overlay_image_base64": image_to_base64(overlay_image),
            "classes": detected_classes,
            "class_colors": {CLASS_NAMES[i]: CLASS_COLORS[i] for i in range(len(CLASS_NAMES))}
        })

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_info = log_error(e, file)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/class-info/")
async def get_class_info():
    """Endpoint to get class names and their colors"""
    color_dict = {}
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        color_dict[name] = {"color": color, "id": i}
    
    return JSONResponse({
        "class_names": CLASS_NAMES,
        "class_colors": color_dict
    })

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "model_loaded": True}