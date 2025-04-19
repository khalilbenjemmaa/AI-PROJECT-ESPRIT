
import streamlit as st
from PIL import Image
import base64
import requests
import time
import pandas as pd
from io import BytesIO

# Configure page settings
st.set_page_config(
    page_title="Marine Debris Segmentation",
    page_icon="🌊",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"

# Helper functions
def decode_image(base64_string):
    """Decode base64 string to PIL Image"""
    decoded_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(decoded_bytes))

def get_class_info():
    """Get class information from the API"""
    try:
        response = requests.get(f"{API_URL}/class-info/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch class information: {e}")
        return None

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .class-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        margin: 2px;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4e83ab, #57a0d3);
    }
</style>
""", unsafe_allow_html=True)

# Check API connection
api_available = check_api_health()

# Header
st.markdown("<div class='main-title'>🌊 Marine Debris Segmentation</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload sonar images to detect and segment underwater debris</div>", unsafe_allow_html=True)

if not api_available:
    st.error("⚠️ API server is not available. Please make sure the FastAPI server is running.")
    st.stop()

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a UNet model to segment underwater debris in sonar images.
    
    Upload an image to see the segmentation results.
    """)
    
    st.header("🏷️ Class Legend")
    try:
        class_info = get_class_info()
        if class_info:
            for name, details in class_info["class_colors"].items():
                if name != "Background":
                    color = details["color"]
                    hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    st.markdown(
                        f"<div class='class-chip' style='background-color: {hex_color};'>{name}</div>", 
                        unsafe_allow_html=True
                    )
    except Exception as e:
        st.error(f"Error loading class information: {e}")

# Main content
uploaded_file = st.file_uploader("📷 Upload a sonar image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Create two columns for original and processed images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    # Process image
    with st.spinner("⏳ Processing image..."):
        try:
            # Create progress bar for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulating processing steps
                time.sleep(0.01)
                progress_bar.progress(i + 1)
                
            # Actual API call
            uploaded_file.seek(0)  # Reset file pointer
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict/", files=files)
            response.raise_for_status()
            result = response.json()
            
            # Display segmentation overlay
            with col2:
                st.subheader("🎯 Segmentation Result")
                overlay_image = decode_image(result["overlay_image_base64"])
                st.image(overlay_image, use_container_width=True)
            
            # Display detected classes with counts
            st.subheader("📊 Detected Classes")
            if not result["classes"]:
                st.info("No debris classes detected in this image.")
            else:
                # Create dataframe for better visualization
                classes_df = pd.DataFrame(
                    [(k, v) for k, v in result["classes"].items()],
                    columns=["Class", "Pixel Count"]
                )
                classes_df = classes_df[classes_df["Class"] != "Background"]
                
                # Sort by pixel count
                classes_df = classes_df.sort_values(by="Pixel Count", ascending=False)
                
                # Display as bar chart
                st.bar_chart(classes_df.set_index("Class"))
                
                # Create class chips with counts
                st.markdown("### Class Distribution")
                for class_name, count in result["classes"].items():
                    if class_name != "Background":
                        color = result["class_colors"][class_name]
                        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                        st.markdown(
                            f"<div class='class-chip' style='background-color: {hex_color};'>{class_name}: {count} pixels</div>", 
                            unsafe_allow_html=True
                        )
            
            # Option to toggle between mask and overlay
            view_option = st.radio(
                "View options", 
                ["Segmentation Overlay", "Segmentation Mask Only"], 
                horizontal=True
            )
            
            if view_option == "Segmentation Mask Only":
                mask_image = decode_image(result["segmentation_map_base64"])
                st.image(mask_image, caption="Segmentation Mask", use_container_width=True)
                
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"API Response: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    # Show sample images or instructions when no file is uploaded
    st.info("👆 Upload an image to see the segmentation results.")
    
    # Show a placeholder or example
    st.markdown("### How it works")
    st.write("""
    1. Upload a sonar image containing marine debris
    2. The model will segment the image and identify debris types
    3. View the results with overlay visualization and class distribution
    """)

# Footer
st.markdown("---")
st.markdown("Marine Debris Segmentation App | Powered by FastAPI and Streamlit")