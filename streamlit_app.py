import streamlit as st
import base64
import numpy as np
import cv2
import os
import gdown
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd




# --------------------------------------------CREATE FUNCTION ------------------------------------------------

# Function to get model id from google drive
def load_from_drive(file_id, output_path):
    """Download file from Google Drive if not already cached locally"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

# Function to get local image as base64
def get_base64_of_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ------------------------------------------- BACKGROUND IMAGE ------------------------------------------------
# Path to local image
image_path = "background.jpg"  # put your file in same folder as app.py
base64_image = get_base64_of_image(image_path)

# Inject CSS
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------------------------------- APPS TITLE ------------------------------------------------
# Title and subtitle
st.title("AINNA 🤖⚙️")
st.subheader("Automated Intelligent Neural Network Architecture")
st.write("**🔍 Detection and Severity Classification of Autism Disorder Using Image-Based Analysis 📊📉**")

# ------------------------------------------- SIDEBAR NAVIGATION ------------------------------------------------
st.sidebar.title("AINNA Navigator 🧭")
page = st.sidebar.radio("",["Home 🏠", "ASD Prediction 🔍", "Performance Dashboard 📊"])

# ------------------------------------------- HOME PAGE ------------------------------------------------
if page == "Home 🏠":
    st.markdown("")
    st.write(
        """
        **Autism Spectrum Disorder (ASD)** is a developmental condition that **affects communication, behavior,
        and learning** in children. **Early detection is crucial** to ensure timely intervention and support.  
        """
    )


# ------------------------------------------- ASD Prediction Page ------------------------------------------------
elif page == "ASD Prediction 🔍":
    st.markdown("### **ASD Prediction**")
    st.write("Select a task type and upload an image of a child’s visual activity to predict ASD severity.")

    # ---- User chooses task ----
    task = st.radio("Select Task", ["Coloring 🎨", "Drawing 🖍️", "Handwriting ✍🏻"])

    # ---- Preprocessing ----
    def preprocess_coloring(image):
        #Preprocessing pipeline for coloring task
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # uploaded image is RGB

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Mask creation
        lower_bound = np.array([0, 20, 50])
        upper_bound = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Fill gaps
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Channel 1: Edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Channel 2: Texture
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Apply mask
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
        texture_masked = cv2.bitwise_and(adaptive_thresh, adaptive_thresh, mask=mask)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

        # Resize & stack
        desired_size = (224, 224)
        edges_resized = cv2.resize(edges_masked, desired_size)
        texture_resized = cv2.resize(texture_masked, desired_size)
        gray_resized = cv2.resize(gray_masked, desired_size)

        final_image = np.stack([edges_resized, texture_resized, gray_resized], axis=-1)
        final_image = final_image / 255.0  # normalize
        final_image = np.expand_dims(final_image, axis=0)  # add batch dimension

        return final_image

    
    def preprocess_image(image):
        #Preprocessing pipeline for drawing and handwriting task
        
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # 1. Reduce noise
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # 2. Otsu threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Morphological opening
        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 4. Resize & stack into 3 channels
        desired_size = (224, 224)
        resized = cv2.resize(opened, desired_size)
        stacked_img = np.stack([resized, resized, resized], axis=-1)

        stacked_img = stacked_img / 255.0  # normalize
        stacked_img = np.expand_dims(stacked_img, axis=0)  # add batch dimension

        return stacked_img

        
    # ---- Map tasks to Google Drive model IDs ----
    model_ids = {
        "Coloring 🎨": {
            "cnn": "1xkO5I_KP9EshFkaZnQ7N5WLuANYz3vrf",
            "svm": "1bE0oS_Hd3e_b9Rx6OJZIJJG6EE2D5qZi"
        },
        "Drawing 🖍️": {
            "cnn": "1mf4LZIfrkn7T9NJ3rZ6TO3de4iQ8-L6E",
            "svm": "1HSRtfhmZJBu7DvBeC_KP9IfxbaZuOAsW"
        },
        "Handwriting ✍🏻": {
            "cnn": "1e2OfwnoDrII-XR6-clNQ90866nVV8tNe",
            "svm": "1sEci79F-qbkUobUmi8GJAerAAKiM5w9C"
        }
        }
    
    preprocess_map = {
    "Coloring 🎨": preprocess_coloring,
    "Drawing 🖍️": preprocess_image,
    "Handwriting ✍🏻": preprocess_image
}
    
    label_classes = np.load("label_encoder.npy", allow_pickle=True)


    # ---- Load correct CNN + SVM models ----
    @st.cache_resource
    def load_models(task):
        cnn_model_full = load_model(load_from_drive(model_ids[task]["cnn"], f"{task}_cnn.h5"))
        cnn_model = Model(inputs=cnn_model_full.input, outputs=cnn_model_full.get_layer("dense_feature_layer").output)
        svm_model = joblib.load(load_from_drive(model_ids[task]["svm"], f"{task}_svm.pkl"))
        return cnn_model, svm_model

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        cnn_model, svm_model = load_models(task)
        
        
        # ---- Prediction function ----
        def predict(img):
            features = cnn_model.predict(img)                  
            prob = svm_model.predict_proba(features)[0]
            pred_idx = prob.argmax()          
            label = label_classes[pred_idx]                    
            return label, prob

    # ---- Predict button ----
        if st.button("🔮 Predict ASD"):
            preprocess_fn = preprocess_map[task]
            processed = preprocess_fn(image)
            label, prob = predict(processed)

            st.success(f"**Predicted ASD Category: {label}**")

            conf_df = pd.DataFrame({
            "Category": label_classes,
            "Confidence": prob * 100
            })

            # Confidence per class
            #conf_dict = {cls: f"{round(p*100,2)}%" for cls, p in zip(label_classes, prob)}
            #st.write("### Confidence Scores:")
            #st.json(conf_dict)

            fig = px.bar(
            conf_df, 
            x="Confidence", 
            y="Category", 
            orientation="h",
            text="Confidence",
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Set2
            )

            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(xaxis_title="Confidence (%)", yaxis_title="ASD Category")

            st.plotly_chart(fig)

            # Suggestions
            if label == "Non_ASD":
                st.info("✅ No ASD detected. Continue monitoring child’s development.")
            elif label == "ASD_Mild":
                st.warning("⚠️ Mild ASD detected. Early behavioral support is recommended.")
            elif label == "ASD_Moderate":
                st.warning("⚠️ Moderate ASD detected. A full clinical evaluation is strongly advised.")
            else:
                st.error("⚠️ Severe ASD detected. Immediate clinical intervention is recommended.")

# ------------------------------------------- Performance Dashboard Page ------------------------------------------------
elif page == "Performance Dashboard 📊":
    st.markdown("### **Performance Dashboard**")
    st.write("FHSDGJSD")
