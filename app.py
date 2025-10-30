import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os

# Try to import TensorFlow/Keras in a safe way
TF_AVAILABLE = True
try:
    from tensorflow.keras.models import load_model
except Exception:
    TF_AVAILABLE = False

# Constants
MODEL_PATH = "brain_tumor_detection_model.h5"

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection AI", 
    page_icon="🧠", 
    layout="centered"
)

# Custom CSS for black & orange theme
st.markdown(
    """
    <style>
    body {
        background-color: #0e0e0e;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #0e0e0e;
    }
    .stButton>button {
        background-color: #ff8c00;
        color: black;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ffa500;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(255, 140, 0, 0.4);
    }
    .upload-box {
        border: 2px dashed #ff8c00;
        padding: 2em;
        border-radius: 10px;
        text-align: center;
        background-color: rgba(255, 165, 0, 0.05);
    }
    .result-box {
        background: linear-gradient(135deg, rgba(255, 140, 0, 0.1), rgba(255, 165, 0, 0.05));
        border: 2px solid #ff8c00;
        border-radius: 10px;
        padding: 1.5em;
        margin-top: 1em;
    }
    h1 {
        color: #ff8c00;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #ff8c00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Brain Tumor Detection System")
st.markdown(
    "<p style='text-align: center; color: #ccc;'>Upload an MRI scan to detect potential brain tumors using AI</p>",
    unsafe_allow_html=True
)
st.markdown("---")


def app_error(msg: str):
    """Display error message and stop execution"""
    st.error(msg)
    st.stop()


@st.cache_resource
def load_model_resource(path: str):
    """Load the Keras model with caching"""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not available. Please install: pip install tensorflow"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess image for model input
    - Convert to RGB
    - Resize to target size
    - Normalize to [0, 1]
    
    Returns: numpy array of shape (1, H, W, C)
    """
    # Ensure RGB format
    image = image.convert("RGB")
    
    # Resize image maintaining aspect ratio
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    arr = np.asarray(image).astype(np.float32) / 255.0
    
    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    
    return arr


# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Model info
with st.sidebar.expander("📊 Model Information", expanded=True):
    st.write(f"**Path:** `{MODEL_PATH}`")
    st.write("**Framework:** TensorFlow/Keras")
    st.write(f"**Status:** {'✅ Available' if TF_AVAILABLE else '❌ Not Available'}")

# Detection threshold
threshold = st.sidebar.slider(
    "Detection Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.01,
    help="Adjust the confidence threshold for tumor detection"
)

# Class labels (customize based on your model)
class_labels = {
    0: "No Tumor",
    1: "Glioma",
    2: "Meningioma",
    3: "Pituitary Tumor"
}

# Check TensorFlow availability
if not TF_AVAILABLE:
    app_error(
        "⚠️ TensorFlow is not installed. Please install it using:\n\n"
        "```bash\npip install tensorflow\n```"
    )

# Load model
model = None
load_error = None

try:
    with st.spinner("Loading model..."):
        model = load_model_resource(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    load_error = e
    st.sidebar.error(f"❌ Model load error: {e}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI scan in JPG or PNG format"
    )

with col2:
    st.markdown("### 📋 Instructions")
    st.markdown(
        """
        1. Upload MRI image
        2. Click **Predict**
        3. View results
        """
    )

st.markdown("---")

if uploaded_file is None:
    st.info("👆 Please upload an MRI image to begin analysis")
else:
    # Display uploaded image
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not open image: {e}")
        st.stop()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # use_column_width is compatible with older Streamlit versions
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Check if model is loaded
    if model is None:
        st.error(f"❌ Model not available: {load_error}")
        st.info("Please ensure the model file exists in the correct location.")
    else:
        # Get target size from model
        try:
            input_shape = model.input_shape
            if len(input_shape) == 4:
                # Format: (None, H, W, C)
                target_size = (input_shape[1], input_shape[2])
            else:
                target_size = (224, 224)
        except Exception:
            target_size = (224, 224)

        st.markdown("---")
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔍 Analyze MRI Scan")

        if predict_button:
            with st.spinner("🧠 Analyzing MRI scan..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image, target_size)
                    
                    # Make prediction
                    predictions = model.predict(processed_image, verbose=0)
                    
                    # Interpret predictions
                    if predictions.shape[1] == 1:
                        # Binary classification
                        prob = float(predictions[0, 0])
                        label = "Tumor Detected" if prob >= threshold else "No Tumor"
                        confidence = prob if prob >= threshold else (1.0 - prob)
                        
                        st.markdown(
                            f"""
                            <div class='result-box'>
                                <h2 style='text-align: center; color: {"#ff4444" if label == "Tumor Detected" else "#44ff44"};'>
                                    {label}
                                </h2>
                                <h3 style='text-align: center;'>
                                    Confidence: {confidence * 100:.2f}%
                                </h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    else:
                        # Multi-class classification
                        class_idx = int(np.argmax(predictions[0]))
                        confidence = float(np.max(predictions[0]))
                        label = class_labels.get(class_idx, f"Class {class_idx}")
                        
                        st.markdown(
                            f"""
                            <div class='result-box'>
                                <h2 style='text-align: center; color: #ff8c00;'>
                                    {label}
                                </h2>
                                <h3 style='text-align: center;'>
                                    Confidence: {confidence * 100:.2f}%
                                </h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Show all class probabilities
                        st.markdown("### 📊 Detailed Predictions")
                        for idx, prob in enumerate(predictions[0]):
                            class_name = class_labels.get(idx, f"Class {idx}")
                            st.progress(float(prob), text=f"{class_name}: {prob * 100:.2f}%")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.warning(
                        "⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only. "
                        "It should NOT be used for actual medical diagnosis. Always consult qualified "
                        "healthcare professionals for medical advice."
                    )
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Powered by Deep Learning | TensorFlow & Keras</p>",
    unsafe_allow_html=True
)