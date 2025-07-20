import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import io

# Set page config
st.set_page_config(
    page_title="BiteNBarbell",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .calorie-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .nutrition-info {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF8C00;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Food database with nutritional information
FOOD_DATABASE = {
    'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fat': 0.3, 'fiber': 4.4},
    'banana': {'calories': 105, 'protein': 1.3, 'carbs': 27, 'fat': 0.4, 'fiber': 3.1},
    'orange': {'calories': 62, 'protein': 1.2, 'carbs': 15, 'fat': 0.2, 'fiber': 3.1},
    'pizza': {'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10, 'fiber': 2.5},
    'hamburger': {'calories': 354, 'protein': 16, 'carbs': 30, 'fat': 17, 'fiber': 2.0},
    'salad': {'calories': 20, 'protein': 2, 'carbs': 4, 'fat': 0.2, 'fiber': 1.5},
    'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0},
    'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4},
    'bread': {'calories': 79, 'protein': 3.1, 'carbs': 15, 'fat': 1, 'fiber': 1.2},
    'milk': {'calories': 103, 'protein': 8, 'carbs': 12, 'fat': 2.4, 'fiber': 0},
    'egg': {'calories': 78, 'protein': 6.3, 'carbs': 0.6, 'fat': 5.3, 'fiber': 0},
    'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 12, 'fiber': 0},
    'steak': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15, 'fiber': 0},
    'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1, 'fiber': 1.8},
    'soup': {'calories': 50, 'protein': 3, 'carbs': 8, 'fat': 1, 'fiber': 1.5},
    'sandwich': {'calories': 300, 'protein': 15, 'carbs': 35, 'fat': 12, 'fiber': 3},
    'cake': {'calories': 257, 'protein': 3.2, 'carbs': 35, 'fat': 12, 'fiber': 0.8},
    'ice_cream': {'calories': 207, 'protein': 3.5, 'carbs': 24, 'fat': 11, 'fiber': 0},
    'coffee': {'calories': 2, 'protein': 0.3, 'carbs': 0, 'fat': 0, 'fiber': 0},
    'tea': {'calories': 1, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0},
}

@st.cache_resource
def load_model():
    """Load the pre-trained MobileNetV2 model"""
    try:
        model = MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    # Ensure image is RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize image to 224x224 (MobileNetV2 input size)
    img = img.resize((224, 224))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess for MobileNetV2
    img_array = preprocess_input(img_array)
    return img_array

def predict_food(model, img_array):
    """Predict food items in the image"""
    try:
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return []

def get_nutrition_info(food_name):
    """Get nutrition information for a food item"""
    # Clean food name and try to match with database
    food_name = food_name.lower().replace('_', ' ').replace('-', ' ')
    
    # Direct matches
    if food_name in FOOD_DATABASE:
        return FOOD_DATABASE[food_name]
    
    # Partial matches
    for key in FOOD_DATABASE.keys():
        if key in food_name or food_name in key:
            return FOOD_DATABASE[key]
    
    # Default nutrition info for unknown foods
    return {'calories': 100, 'protein': 5, 'carbs': 15, 'fat': 3, 'fiber': 2}

def create_nutrition_chart(nutrition_data):
    """Create a bar chart for nutrition information"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    nutrients = list(nutrition_data.keys())
    values = list(nutrition_data.values())
    
    bars = ax.bar(nutrients, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax.set_title('Nutritional Information', fontsize=16, fontweight='bold')
    ax.set_ylabel('Amount', fontsize=12)
    ax.set_xlabel('Nutrients', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏋️‍♂️ BiteNBarbell</h1>', unsafe_allow_html=True)
    st.markdown("### Upload a food image and get instant nutritional analysis!")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the AI model. Please check your internet connection.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of food to analyze its nutritional content"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Your food image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 AI Analysis")
            
            # Preprocess image
            with st.spinner("Analyzing image..."):
                img_array = preprocess_image(image_display)
                predictions = predict_food(model, img_array)
            
            if predictions:
                st.success("✅ Analysis complete!")
                
                # Display top predictions
                st.write("**Top predictions:**")
                for i, (imagenet_id, label, score) in enumerate(predictions):
                    st.write(f"{i+1}. {label.replace('_', ' ').title()} ({score:.2%})")
                
                # Get nutrition info for top prediction
                top_food = predictions[0][1]
                nutrition_info = get_nutrition_info(top_food)
                
                # Display nutrition information
                st.markdown('<div class="calorie-box">', unsafe_allow_html=True)
                st.markdown(f"### 🍽️ **{top_food.replace('_', ' ').title()}**")
                st.markdown(f"**Estimated Calories:** {nutrition_info['calories']} kcal")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="nutrition-info">', unsafe_allow_html=True)
                st.markdown("### 📊 **Nutritional Breakdown (per serving):**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Protein", f"{nutrition_info['protein']}g")
                    st.metric("Carbs", f"{nutrition_info['carbs']}g")
                
                with col2:
                    st.metric("Fat", f"{nutrition_info['fat']}g")
                    st.metric("Fiber", f"{nutrition_info['fiber']}g")
                
                with col3:
                    st.metric("Calories", f"{nutrition_info['calories']} kcal")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Create and display nutrition chart
                st.subheader("�� Nutrition Chart")
                fig = create_nutrition_chart(nutrition_info)
                st.pyplot(fig)
                
                # Additional insights
                st.subheader("�� Health Insights")
                calories = nutrition_info['calories']
                
                if calories < 100:
                    st.info("🍃 This is a low-calorie food item, great for weight management!")
                elif calories < 300:
                    st.success("✅ This is a moderate-calorie food item, suitable for a balanced diet.")
                else:
                    st.warning("⚠️ This is a high-calorie food item. Consider portion control.")
                
                # Daily value percentages (based on 2000 calorie diet)
                st.subheader("📋 Daily Value (%DV)")
                daily_values = {
                    'Calories': (calories / 2000) * 100,
                    'Protein': (nutrition_info['protein'] / 50) * 100,
                    'Carbs': (nutrition_info['carbs'] / 275) * 100,
                    'Fat': (nutrition_info['fat'] / 65) * 100,
                    'Fiber': (nutrition_info['fiber'] / 28) * 100
                }
                
                for nutrient, percentage in daily_values.items():
                    st.progress(min(percentage / 100, 1.0))
                    st.write(f"{nutrient}: {percentage:.1f}% of daily value")
                
            else:
                st.error("❌ Could not analyze the image. Please try with a clearer food image.")
    
    # Sidebar with additional features
    with st.sidebar:
        st.header(" About")
        st.write("""
        This AI-powered nutrition analyzer uses deep learning to:
        - Identify food items in images
        - Calculate estimated calories
        - Provide nutritional breakdown
        - Give health insights
        
        **How to use:**
        1. Upload a clear food image
        2. Wait for AI analysis
        3. Review nutritional information
        4. Get health insights
        """)
        
        st.header("Features")
        st.write("""
        ✅ Food recognition
        ✅ Calorie estimation
        ✅ Nutritional breakdown
        ✅ Health insights
        ✅ Visual charts
        ✅ Daily value percentages
        """)
        
        st.header("🔧 Technical Info")
        st.write("""
        **Model:** MobileNetV2
        **Framework:** TensorFlow
        **UI:** Streamlit
        """)

if __name__ == "__main__":
    main()
    