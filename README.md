[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neuroaina.streamlit.app/)

# 🧠 EARLY DIAGNOSIS OF AUTISM SPECTRUM DISORDER USING CONVOLUTIONAL NEURAL NETWORK WITH DASHBOARD VISUALIZATION
This project is a web-based system that classifies children’s visual tasks (coloring, drawing, and handwriting) to detect potential signs of Autism Spectrum Disorder (ASD). It uses deep learning models, CNN for feature extraction, followed by an SVM classifier for final classification. The interface is built with Streamlit and includes a dashboard visualization for data insights and analysis.

# 🚀 Features
- Classifies uploaded images into 4 ASD categories: Non-ASD, Mild, Moderate, Severe.
- Dashboard visualization with prediction result and analysis
- Simple web interface with no installation required when deployed via Streamlit Cloud

  # 🧪 Model Training
- Feature extractors: CNNs
- Classifier: Support Vector Machine (Linear Kernel)
- Dataset: Labeled ASD visual tasks (handwriting, drawing, coloring)

  # 🙏 Acknowledgments
- Project developed for Master in Technology (Data Science and Analytics), Universiti Teknikal Malaysia Melaka (UTeM)
- Data inspired by publicly available ASD handwriting/drawing datasets and validated by an occupational therapist from the Centre for Rehabilitation & Special Needs Studies (iCaRehab), Faculty of Health Sciences, Universiti Kebangsaan Malaysia, Kuala Lumpur
- Dataset source: https://www.kaggle.com/datasets/imranliaqat32/autism-spectrum-disorder-in-childrenhandgestures

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
