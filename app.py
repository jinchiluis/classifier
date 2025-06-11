import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import json
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="Face Classifier",
    page_icon="👤",
    layout="centered"
)

@st.cache_resource
def load_model_and_indices():
    """Load the model and class indices"""
    try:
        model = tf.keras.models.load_model('face_classifier_model.h5')
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
        return model, class_indices, class_names
    except Exception as e:
        st.error(f"Error loading model or indices: {str(e)}")
        return None, None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model input"""
    # Convert PIL image to array
    img = image.resize(target_size)
    img_array = np.array(img)
    
    # Normalize to 0-1 range
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def _find_last_conv(model):
    """
    Returns the last convolution-type layer inside `model`.
    Works whether the first child is a nested base_model (e.g. MobileNetV2)
    or the conv layers are top-level.
    """
    # if model.layers[0] is itself a Model, search inside it first
    search_layers = (
        model.layers[0].layers
        if isinstance(model.layers[0], tf.keras.Model)
        else model.layers
    )

    for lyr in reversed(search_layers):
        if isinstance(
            lyr, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
        ):
            return lyr
    return None


# ──────────────────────────────────────────────────────────────
# main: robust Grad-CAM
# ──────────────────────────────────────────────────────────────
def generate_gradcam(model, img_array, class_idx, debug=False):
    """
    Grad-CAM that works for a Sequential model whose first layer is a
    transfer-learning backbone (e.g. MobileNetV2) followed by a custom head.

    Parameters
    ----------
    model      : tf.keras.Sequential      your face-classifier
    img_array  : np.ndarray (1, H, W, 3)  pre-processed image
    class_idx  : int                      target class index (0 or 1)
    debug      : bool                     True ⇒ show shapes in sidebar
    """
    # ------------------------------------------------------------------
    # 1  Split the architecture into backbone and head
    # ------------------------------------------------------------------
    if isinstance(model.layers[0], tf.keras.Model):
        backbone    = model.layers[0]      # MobileNetV2
        head_layers = model.layers[1:]     # GAP, BN, Dense, …
    else:                                  # very rare, but keep it safe
        backbone    = model
        head_layers = []

    # ------------------------------------------------------------------
    # 2  Build ONE fresh functional graph
    # ------------------------------------------------------------------
    inp        = tf.keras.Input(shape=model.input_shape[1:], name="gradcam_in")
    conv_feat  = backbone(inp, training=False)          # 7×7×1280 feature map
    x          = conv_feat
    for lyr in head_layers:                             # run the head manually
        x = lyr(x, training=False)
    preds      = x                                      # final logits / softmax

    grad_model = tf.keras.Model(inputs=inp,
                                outputs=[conv_feat, preds],
                                name="gradcam_model")

    if debug:  # optional peek in the Streamlit sidebar
        st.sidebar.markdown("### Grad-CAM debug")
        st.sidebar.write({
            "conv_feature_map": conv_feat.shape,
            "preds_shape"     : preds.shape,
        })

    # ------------------------------------------------------------------
    # 3  Standard Grad-CAM math
    # ------------------------------------------------------------------
    with tf.GradientTape() as tape:
        conv_vals, preds_vals = grad_model(img_array, training=False)
        loss = preds_vals[:, class_idx]

    grads = tape.gradient(loss, conv_vals)
    if grads is None:
        st.error("📛  Gradient is None – Grad-CAM aborted.")
        return np.zeros((7, 7))

    pooled_grads   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_vals      = conv_vals[0]              # drop batch dimension
    conv_vals     *= pooled_grads              # weight each channel
    heatmap        = tf.reduce_mean(conv_vals, axis=-1).numpy()

    # normalise to 0-1 for display
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1.0
    return heatmap

def create_gradcam_visualization(image, heatmap):
    """Create Grad-CAM visualization overlay"""
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
    
    # Convert to RGB
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Convert PIL image to array
    img_array = np.array(image)
    
    # Create overlay
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay, heatmap_colored

# Main app
st.title("🎯 Face Classifier with Grad-CAM")
st.markdown("Upload an image to classify which person it is!")

# Load model and indices
model, class_indices, class_names = load_model_and_indices()

if model is None:
    st.error("Please ensure 'face_classifier_model.h5' and 'class_indices.json' are in the same directory as this script.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Add GO button
    if st.button("🚀 GO - Classify!", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            # Preprocess image
            img_array = preprocess_image(image)
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            person_name = class_names[predicted_class]
            
            # Generate Grad-CAM
            try:
                heatmap = generate_gradcam(model, img_array, predicted_class, debug=True)
                overlay, heatmap_colored = create_gradcam_visualization(image, heatmap)
                gradcam_available = True
            except Exception as e:
                st.warning(f"Could not generate Grad-CAM visualization: {str(e)}")
                gradcam_available = False
                heatmap = None
            
            # Display results
            st.success(f"**Prediction: {person_name}** (Confidence: {confidence:.2%})")
            
            # Display Grad-CAM
            with col2:
                st.subheader("Grad-CAM Heatmap")
                if gradcam_available:
                    st.image(overlay, use_column_width=True)
                else:
                    st.info("Grad-CAM visualization not available for this model architecture")
            
            # Additional details
            with st.expander("📊 Detailed Results"):
                # Prediction probabilities
                st.subheader("Prediction Probabilities")
                for idx, prob in enumerate(predictions[0]):
                    person = class_names[idx]
                    # Convert float32 to Python float
                    prob_value = float(prob)
                    st.progress(prob_value)
                    st.text(f"{person}: {prob_value:.2%}")
                
                # Display raw heatmap
                if gradcam_available and heatmap is not None:
                    st.subheader("Raw Heatmap")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(heatmap, cmap='jet')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                    plt.close()

# Add information section
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### How it works:
    1. Upload an image of a face
    2. Click the **GO** button
    3. The model will classify which person it is
    4. Grad-CAM shows which parts of the image the model focused on
    
    ### Grad-CAM Colors:
    - 🔴 Red: High importance
    - 🟡 Yellow: Medium importance
    - 🔵 Blue: Low importance
    """)
    
    if class_names:
        st.subheader("📋 Available Classes")
        for idx, name in class_names.items():
            st.write(f"- {name}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and TensorFlow")