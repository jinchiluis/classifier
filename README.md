# Face Classifier

This project contains a small convolutional neural network for classifying two different faces along with a Streamlit web interface and some helper utilities.

## Setup

1. Create a virtual environment (optional but recommended).
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train or fine tune the model use `improved_face_classifier.py`:

```bash
python improved_face_classifier.py
```

### Dataset analysis

```bash
python analyzer.py
```

### Running the web app

```bash
streamlit run app.py
```

The application loads the trained model and `class_indices.json` to make predictions on uploaded images.

## Repository contents

- `improved_face_classifier.py` – training script built with TensorFlow/Keras.
- `analyzer.py` – utilities to inspect and visualize the dataset.
- `app.py` – Streamlit interface for performing predictions.
- `class_indices.json` – mapping of class names to indices.
- `requirements.txt` – list of Python dependencies.
