# Student Attention Detection App

A real time attention detection application that uses a TensorFlow image-classification model to identify whether a student is **Attentive** or **Not Attentive**. Images can be uploaded or captured with a webcam, and the app displays the prediction confidence.

## Features

- Upload JPG, JPEG, or PNG images for analysis
- Capture images using a webcam
- Classify student engagement into two categories
- Display prediction confidence and color-coded results

## Requirements

- Python 3.10
- A trained model named `student_engagement_model.h5` in the project root

## Installation

```bash
git clone https://github.com/abdullahhimel46/Student-attention-detection-app.git
cd Student-attention-detection-app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, then upload an image or use the webcam to analyze student attention.

## Project Structure

```text
app.py              # Streamlit application and prediction logic
requirements.txt    # Python dependencies
runtime.txt         # Python runtime version
templates/          # Alternative HTML templates
```

## Notes

The application resizes input images to `224 × 224` pixels and normalizes pixel values before inference. Prediction quality depends on the training data and the supplied TensorFlow model.
