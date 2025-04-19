import os
import io
import torch
import gdown
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT_PATH = "trained_modal/epoch_30.pt"
GDRIVE_FILE_ID = "1DeTPtUEZI9b1C9f4OsTzGCk-febyUCvW"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Auto-download model if not found using gdown
if not os.path.exists(CHECKPOINT_PATH):
    print("⬇️ Téléchargement du modèle depuis Google Drive avec gdown...")
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    gdown.download(GDRIVE_URL, CHECKPOINT_PATH, quiet=False)
    print("✅ Modèle téléchargé avec succès.")

# Classes et couleurs
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

# Load model
try:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=1, out_channels=12, init_features=64, pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
    raise

# Transformations d'image
transform = T.Compose([
    T.Grayscale(),
    T.Resize((256, 256)),
    T.ToTensor()
])

# Logger d'erreur
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
    print("\n🔴 Détails de l'erreur :")
    for k, v in error_details.items():
        print(f"│ {k.upper():<15}: {v}")
    return error_details

# Overlay mask
def create_overlay(original_image, segmentation_mask, alpha=0.5):
    mask_resized = Image.fromarray(segmentation_mask).resize(original_image.size, Image.NEAREST)
    overlay = Image.blend(original_image.convert('RGB'), mask_resized, alpha)
    return overlay

# Endpoint de prédiction
@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file.file = io.BytesIO(contents)

        try:
            original_image = Image.open(file.file)
            original_image.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Fichier image invalide : {str(e)}")

        file.file.seek(0)
        original_image = Image.open(file.file).convert("RGB")
        original_image_copy = original_image.copy()

        tensor_img = transform(original_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            predicted = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        present_classes = np.unique(predicted).tolist()
        class_count = {CLASS_NAMES[idx]: np.sum(predicted == idx) for idx in present_classes}
        detected_classes = {
            CLASS_NAMES[idx]: int(class_count[CLASS_NAMES[idx]])
            for idx in present_classes if idx != 0
        }

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
        raise HTTPException(status_code=500, detail=f"Échec de la prédiction : {str(e)}")

# Endpoint info des classes
@app.get("/class-info/")
async def get_class_info():
    color_dict = {
        name: {"color": color, "id": i}
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
    }
    return JSONResponse({
        "class_names": CLASS_NAMES,
        "class_colors": color_dict
    })

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}
