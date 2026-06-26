import os
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Setup the Streamlit page layout
st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it look nice and modern
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #00e676;
        text-align: center;
        padding: 20px;
        background: rgba(0, 230, 118, 0.1);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #00e676;
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        border: None;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00c853;
        transform: scale(1.02);
    }
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        color: #00e676;
    }
</style>
""", unsafe_allow_html=True)


# Function to load the saved model from disk
@st.cache_resource
def load_saved_model():
    try:
        model = joblib.load('saved_models/rf_model.pkl')
        return model
    except Exception as e:
        # Return None if model is not found
        return None

# Function to extract visual features using SIFT
def extract_sift_features(image):
    # Convert to grayscale first
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    # Return average descriptors
    if descriptors is not None:
        mean_features = np.mean(descriptors, axis=0)
        return mean_features.ravel()
    else:
        return np.zeros(128)

# Function to process the 4 uploaded images and combine them
def process_uploaded_images(uploaded_files, size=256):
    # If the user didn't upload exactly 4 images, return None
    if len(uploaded_files) != 4:
        return None
    
    loaded_images = []
    
    # Read each uploaded file
    for file in uploaded_files:
        # Open image using PIL
        image = Image.open(file).convert('RGB')
        # Convert to numpy array for OpenCV
        image = np.array(image)
        # Convert colors from RGB to BGR (OpenCV format)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Resize to half the target size
        image = cv2.resize(image, (size // 2, size // 2))
        loaded_images.append(image)
        
    # Create an empty black image to hold all 4 images
    combined_image = np.zeros((size, size, 3), dtype='uint8')
    
    # Place the 4 images in the 4 corners
    combined_image[0:size // 2, 0:size // 2] = loaded_images[0]
    combined_image[0:size // 2, size // 2:size] = loaded_images[1]
    combined_image[size // 2:size, size // 2:size] = loaded_images[2]
    combined_image[size // 2:size, 0:size // 2] = loaded_images[3]
    
    # Extract and return the SIFT features for this combined image
    return extract_sift_features(combined_image)


def main():
    # Title and description
    st.markdown('<h1 class="main-header">🏡 Multi-Modal House Price Estimator</h1>', unsafe_allow_html=True)
    st.markdown("Estimate the price of a house using both **Textual Features** and **Visual Images** powered by a Random Forest machine learning model.")

    # Load the machine learning model
    model = load_saved_model()

    # Sidebar for numerical inputs (bedrooms, bathrooms, area)
    with st.sidebar:
        st.header("📋 Textual Features")
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=15.0, value=2.0, step=0.5)
        area = st.number_input("Area (sq ft)", min_value=100, max_value=20000, value=1500, step=100)
        
        st.markdown("---")
        st.markdown("### Ready?")
        estimate_btn = st.button("🔮 Estimate Price")

    # Main area for uploading images
    st.header("📸 Visual Features")
    st.markdown("Please upload **exactly 4 images** of the house (e.g., Front, Bedroom, Bathroom, Kitchen).")
    
    # Create two columns for neat layout
    col1, col2 = st.columns(2)
    
    with col1:
        img1 = st.file_uploader("Upload Image 1 (Front)", type=["jpg", "png", "jpeg"])
        img2 = st.file_uploader("Upload Image 2 (Bedroom)", type=["jpg", "png", "jpeg"])
        
    with col2:
        img3 = st.file_uploader("Upload Image 3 (Bathroom)", type=["jpg", "png", "jpeg"])
        img4 = st.file_uploader("Upload Image 4 (Kitchen)", type=["jpg", "png", "jpeg"])
        
    uploaded_files = [img1, img2, img3, img4]
    
    # When the user clicks the "Estimate Price" button
    if estimate_btn:
        # Check if model loaded successfully
        if model is None:
            st.error("Model not found! Please run train_rf.py to train the model first.")
            return
            
        # Check if they uploaded all 4 images
        if any(img is None for img in uploaded_files):
            st.warning("⚠️ Please upload all 4 images to get an estimation.")
            return
            
        # Show a loading spinner while processing
        with st.spinner("🤖 Extracting features and predicting price..."):
            
            # 1. Get Visual Features (SIFT)
            sift_features = process_uploaded_images(uploaded_files)
            
            # 2. Get Text Features
            text_features = np.array([bedrooms, bathrooms, area])
            
            # 3. Combine both features into one big array
            X_input = np.hstack([sift_features, text_features]).reshape(1, -1)
            
            # 4. Predict the price using the Random Forest model
            predicted_price_usd = model.predict(X_input)[0]
            
            # Simple conversion rate (1 USD = ~83.5 INR)
            predicted_price_inr = predicted_price_usd * 83.5
            
            # 5. Display the final price to the user in both currencies
            st.success("Estimation Complete!")
            
            # Create two columns for the metrics
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Estimated Price (USD)", value=f"${predicted_price_usd:,.2f}")
            with res_col2:
                st.metric(label="Estimated Price (INR)", value=f"₹{predicted_price_inr:,.2f}")
                
            st.balloons()

if __name__ == '__main__':
    main()
