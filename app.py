import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load your trained model
model = tf.keras.models.load_model('skin_disease_classifier4.keras')

# Define image size
img_height, img_width = 224, 224

# Define class names (update this based on your training directory structure)
class_names = [
    'cellulitis', 'impetigo', 'athlete-foot', 'nail-fungus',
    'ringworm', 'cutaneous-larva-migrans', 'chickenpox', 'shingles'
]

# Define causes for each class
disease_causes = {
    'cellulitis': 'Often caused by cuts or wounds exposed to bacteria in soil or contaminated water during fieldwork.',
    'impetigo': 'Can result from minor skin injuries or insect bites in hot, humid farm environments.',
    'athlete-foot': 'Due to prolonged wearing of damp, muddy footwear common in irrigation or wet fieldwork.',
    'nail-fungus': 'Fungal infection from repeated exposure to moist soil or water during farming tasks.',
    'ringworm': 'Highly contagious fungal infection from animals or shared tools and clothing on farms.',
    'cutaneous-larva-migrans': 'Caused by skin contact with soil contaminated with animal feces containing hookworm larvae.',
    'chickenpox': 'Viral infection that can spread quickly in rural communities with limited vaccination coverage.',
    'shingles': 'Reactivation of chickenpox virus, often triggered by physical stress or prolonged sun exposure on the farm.'
}


# Streamlit UI
st.set_page_config(page_title="AgriDerm - Skin Disease Detector", layout="centered")
st.title("🌾 AgriDerm: Skin Disease Detection for Farmers")
st.write("Upload a skin image **or use your camera** to detect potential diseases and view likely causes.")

# Input method
upload_option = st.radio("Choose input method:", ("📤 Upload Image", "📷 Use Camera"))

image_data = None

# Handle upload
if upload_option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = Image.open(uploaded_file).convert("RGB")

# Handle camera input
elif upload_option == "📷 Use Camera":
    captured_image = st.camera_input("Take a photo using your webcam")
    if captured_image:
        image_data = Image.open(io.BytesIO(captured_image.getvalue())).convert("RGB")

# Process image and make prediction
if image_data:
    st.image(image_data, caption="Input Image", use_column_width=True)

    # Preprocess the image
    img = image_data.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    # Show result
    st.success(f"🩺 **Predicted Disease**: {predicted_class}")
    st.info(f"🔍 **Possible Cause**: {disease_causes.get(predicted_class, 'Information not available.')}")
    st.write(f"📊 **Confidence**: {confidence:.2f}%")
