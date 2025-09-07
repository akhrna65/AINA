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
import seaborn as sns
import matplotlib.pyplot as plt

if "prediction_counts" not in st.session_state:
    st.session_state.prediction_counts = {
        "Non_ASD": 0,
        "ASD_Mild": 0,
        "ASD_Moderate": 0,
        "ASD_Severe": 0
    }

if "task_counts" not in st.session_state:
    st.session_state.task_counts = {
        "Coloring 🎨": 0,
        "Drawing 🖍️": 0,
        "Handwriting ✍🏻": 0
    }

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
page = st.sidebar.radio("",["Home 🏠", "ASD Prediction 🔍", "Insight Dashboard 📊", "Model Performance 📈"])

# ------------------------------------------- HOME PAGE ------------------------------------------------
if page == "Home 🏠":
    st.markdown("--------------------------------------------")
    st.write(
        """
        ### About Autism Spectrum Disorder (ASD)
        Autism Spectrum Disorder (ASD) is a **neurodevelopmental condition** that affects a child’s 
        **social interaction, communication, behavior, and learning patterns**.  
        The severity of ASD varies from **mild to severe**, and early detection is critical to provide 
        timely intervention, support, and therapy for improved development outcomes.  

        ### About This Project
        **AINNA (Automated Intelligent Neural Network Architecture)**, is designed to assist 
        in the **early detection and severity classification of ASD** using children’s **visual tasks**:  
        - 🎨 **Coloring**  
        - 🖍️ **Drawing**  
        - ✍🏻 **Handwriting**

        By applying a **hybrid deep learning and machine learning approach** (CNN feature extraction combined with 
        classifiers such as SVM, KNN, and RF), this project aims to:  
        - Detect the **occurrence of ASD** in children.  
        - Predict the **severity level** (Non-ASD, Mild, Moderate, Severe).  
        - Provide insights that may support **parents, educators, and healthcare professionals** in making 
          informed decisions.  

        ### Objectives
        - ✅ To develop an image-based ASD detection and classification system.  
        - ✅ To evaluate and compare hybrid model performance across different visual tasks.  
        - ✅ To provide an interactive dashboard for real-time prediction and insights visualization.  
        """
    )


# ------------------------------------------- ASD Prediction Page ------------------------------------------------
elif page == "ASD Prediction 🔍":
    st.markdown("### **ASD Prediction**")
    st.write("Upload an image of a child’s visual activity to predict ASD severity.")

    # --- Demographic inputs ---
    child_gender = st.selectbox("Select Child Gender", ["Male", "Female"])
    child_age = st.number_input("Enter Child Age", min_value=1, max_value=20, step=1)

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

            #Update Counters After Prediction
            if label in st.session_state.prediction_counts:
                st.session_state.prediction_counts[label] += 1

            # Update task count
            if task in st.session_state.task_counts:
                st.session_state.task_counts[task] += 1

            # Build pie chart dataframe
            counts = st.session_state.prediction_counts
            df = pd.DataFrame({
                "Category": list(counts.keys()),
                "Count": list(counts.values())
            })

            if df["Count"].sum() > 0:
                fig = px.pie(
                    df,
                    values="Count",
                    names="Category",
                    hole=0.4,
                    title="Live ASD Prediction Distribution"
                )
                st.plotly_chart(fig)

            if "history" not in st.session_state:
                st.session_state["history"] = []

            st.session_state["history"].append({
                "gender": child_gender,
                "age": child_age,
                "task": task,
                "prediction": label,  # predicted ASD class
            })

            st.success(f"Prediction: {label}")


# ------------------------------------------- Insight Dashboard Page ------------------------------------------------
elif page == "Insight Dashboard 📊":
    st.markdown("### **Insight Dashboard**")

    if "history" in st.session_state and st.session_state["history"]:
        df = pd.DataFrame(st.session_state["history"])

        # --- Prediction Distribution ---
        pred_counts = df["prediction"].value_counts().reset_index()
        pred_counts.columns = ["Category", "Count"]
        fig1 = px.pie(pred_counts, names="Category", values="Count", title="Prediction Distribution")
        st.plotly_chart(fig1)

        # --- Task Distribution ---
        task_counts = df["task"].value_counts().reset_index()
        task_counts.columns = ["Task", "Count"]
        fig2 = px.bar(task_counts, x="Task", y="Count", color="Task", text="Count",
                      title="Predictions by Task Type")
        st.plotly_chart(fig2)

        # --- Gender Distribution ---
        gender_counts = df["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig3 = px.pie(gender_counts, names="Gender", values="Count", title="Predictions by Gender")
        st.plotly_chart(fig3)

        # --- Gender vs Prediction Breakdown ---
        gender_pred_counts = df.groupby(["gender", "prediction"]).size().reset_index(name="Count")

        fig3 = px.bar(
            gender_pred_counts,
            x="gender",
            y="Count",
            color="prediction",
            barmode="group",
            text="Count",
            title="Predictions by Gender and Category"
        )
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3)

        # --- Age Distribution ---
        fig4 = px.histogram(df, x="age", color="prediction", barmode="group",
                            title="Age Distribution of Predictions")
        st.plotly_chart(fig4)

    else:
        st.info("No predictions yet. Upload an image and make a prediction first.")




# ------------------------------------------- Model Performance Page ------------------------------------------------
elif page == "Model Performance 📈":
    st.markdown("### **Model Performance Summary**")
    st.write("")

    # ------------------ Coloring ------------------
    coloring = {
        "Model": ["CNN+SVM", "CNN+KNN", "CNN+RF", "CNN"],
        "Accuracy": [91, 87, 83, 78],
        "Precision": [94, 87, 82,78],
        "Recall": [92, 88, 82,78],
        "F1-score": [91, 87, 81,76]
    }

    df = pd.DataFrame(coloring)

    # Show table
    st.write("##### Coloring")
    st.dataframe(df)

    # ------------------ Drawing ------------------
    drawing = {
        "Model": ["CNN+SVM", "CNN+KNN", "CNN+RF", "CNN"],
        "Accuracy": [71, 67, 71, 67],
        "Precision": [77, 69, 75,69],
        "Recall": [71, 67, 71,67],
        "F1-score": [71, 67, 70,67]
    }

    df = pd.DataFrame(drawing)

    # Show table
    st.write("##### Drawing")
    st.dataframe(df)

    # ------------------ Handwriting ------------------
    handwriting = {
        "Model": ["CNN+SVM", "CNN+KNN", "CNN+RF", "CNN"],
        "Accuracy": [74, 74, 70, 74],
        "Precision": [75, 84, 76,77],
        "Recall": [74, 74, 70,74],
        "F1-score": [74, 73, 71,75]
    }

    df = pd.DataFrame(handwriting)

    # Show table
    st.write("##### Handwriting")
    st.dataframe(df)

    #Accuracy Comparison by Task & Model
    # Combine all into one DataFrame for visualization
    all_data = {
        "Task": ["Coloring"]*4 + ["Drawing"]*4 + ["Handwriting"]*4,
        "Model": ["CNN+SVM", "CNN+KNN", "CNN+RF", "CNN"]*3,
        "Accuracy": [91, 87, 83, 78,
                    71, 67, 71, 67,
                    74, 74, 70, 74]
    }

    df_all = pd.DataFrame(all_data)

    st.write("#### Accuracy Comparison Across Tasks")
    fig = px.bar(
        df_all,
        x="Task",
        y="Accuracy",
        color="Model",
        barmode="group",
        text="Accuracy",
        title="Model Accuracy by Task"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig)










