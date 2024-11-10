import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model from the .h5 file
model = tf.keras.models.load_model("C:/Users/laksh/Downloads/eye_disease_model.keras.h5")


# Class names for the disease prediction
classes = ['Normal', 'Glaucoma', 'Diabetic Retinopathy', 'Cataract']

# Preprocess the uploaded image to match model input requirements
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust to the model's input size
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app design
st.set_page_config(page_title="Eye Disease Classifier", page_icon=":eye:", layout="centered")

# Title and description
st.title("👁️ Eye Disease Classification")
st.write(
    """
    This application classifies eye images into four categories: **Normal**, **Glaucoma**, 
    **Diabetic Retinopathy**, and **Cataract**. Upload an image to get a prediction.
    """
)

# File uploader
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an eye image", type=["jpg", "jpeg", "png"])

# Main application logic
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=500)

    # Predict button
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            result = classes[np.argmax(prediction)]
            
       
        st.success(f"**Prediction:** {result}")
        
        # st.subheader("Prediction Confidence")
        # confidence_scores = {cls: f"{score*100:.2f}%" for cls, score in zip(classes, prediction[0])}
        # st.json(confidence_scores)
else:
    st.info("👈 Please upload an image to get started!")

# Add styling with CSS for a better look and feel
st.markdown(
    """
    <style>
    .css-1cpxqw2 { text-align: center; }
    .stButton>button { border-radius: 8px; color: white; background-color: #4CAF50; }
    .css-2trqyj { border-radius: 5px; }
    .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)
