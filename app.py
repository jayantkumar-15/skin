import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('C:/Users/91814/Desktop/skin/vs/skin_disease_classifier4.keras')

# Image size expected by the model
img_height, img_width = 224, 224

# Class names
class_names = [
    'cellulitis', 'impetigo', 'athlete-foot', 'nail-fungus',
    'ringworm', 'cutaneous-larva-migrans', 'chickenpox', 'shingles'
]

# Disease causes
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

# UI setup
st.set_page_config(page_title="AgriDerm", layout="centered")
st.title("🌾 AgriDerm: Skin Disease Detection for Farmers")
st.write("Upload a skin image **or use your camera** to detect potential diseases and see possible causes.")

# Input option
upload_option = st.radio("Choose input method:", ("📤 Upload Image", "📷 Use Camera"))

image_data = None
image_filename = None
image_bytes = None

# Upload image
if upload_option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = Image.open(uploaded_file).convert("RGB")
        image_filename = uploaded_file.name
        image_bytes = uploaded_file.getvalue()

# Capture image
elif upload_option == "📷 Use Camera":
    captured_image = st.camera_input("Take a photo using your webcam")
    if captured_image:
        image_data = Image.open(io.BytesIO(captured_image.getvalue())).convert("RGB")
        image_filename = "camera_capture.jpg"
        image_bytes = captured_image.getvalue()

# Display and process image
if image_data:
    st.image(image_data, caption="Input Image", use_column_width=True)
    st.write(f"📝 **Image Filename**: `{image_filename}`")

    # Download button
    if image_bytes:
        st.download_button(
            label="📥 Download Image",
            data=image_bytes,
            file_name=image_filename,
            mime="image/jpeg"
        )

    # Preprocess and predict
    img = image_data.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    # Results
    st.success(f"🩺 **Predicted Disease**: {predicted_class}")
    st.info(f"🔍 **Possible Cause**: {disease_causes.get(predicted_class, 'Information not available.')}")
    st.write(f"📊 **Confidence**: {confidence:.2f}%")


