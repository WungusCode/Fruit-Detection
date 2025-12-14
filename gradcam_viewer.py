"""
============================================================
CPSC 483 – Fruit Freshness Detection Project
GRAD-CAM + MOLD COVERAGE SCRIPT (FULLY COMMENTED)
------------------------------------------------------------
This script:
 - Loads the trained CNN model
 - Processes a single fruit image
 - Generates a Grad-CAM heatmap showing where mold was detected
 - Calculates mold severity as a % of the fruit surface
 - Saves the visualization as gradcam_result.jpg
============================================================
"""

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import cv2
import sys
import os

IMG_SIZE = 150   # Same size used during model training

# ============================================================
# GRAD-CAM + MOLD DETECTION FUNCTION
# ============================================================

def gradcam_with_mold_detection(model, img_path, layer_name="conv3"):

    # Ensure file exists
    if not os.path.exists(img_path):
        print("ERROR: File not found:", img_path)
        return

    print("\nRunning Grad-CAM on:", img_path)

    # -----------------------------------------
    # 1. Load and preprocess the image
    # -----------------------------------------
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    
    # Convert numpy array → TensorFlow tensor and normalize (0–1)
    img_tensor = tf.convert_to_tensor(img_arr[np.newaxis, ...] / 255.0)

    # -----------------------------------------
    # 2. Get the layer used for Grad-CAM
    # -----------------------------------------
    try:
        conv_layer = model.get_layer(layer_name)
    except:
        print("ERROR: Layer", layer_name, "not found.")
        return

    # Build a model that outputs:
    # - The chosen convolutional layer
    # - The final prediction
    grad_model = Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    # -----------------------------------------
    # 3. Compute gradients (Grad-CAM core)
    # -----------------------------------------
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_tensor)
        tape.watch(conv_output)              # Watch activations for gradient use
        class_score = prediction[:, 0]       # Rotten class probability

    grads = tape.gradient(class_score, conv_output)

    if grads is None:
        print("ERROR: Gradients could not be computed.")
        return

    # Convert tensors to numpy arrays
    conv_output = conv_output[0].numpy()
    grads = grads[0].numpy()

    # -----------------------------------------
    # 4. Compute channel weights (importance)
    # -----------------------------------------
    weights = np.mean(grads, axis=(0, 1))   # Importance of each filter

    # Build CAM by weighting each filter activation
    cam = np.zeros(conv_output.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    # ReLU: keep only positive values
    cam = np.maximum(cam, 0)

    # Normalize CAM to (0–1)
    if cam.max() != 0:
        cam /= cam.max()

    # -----------------------------------------
    # 5. Mold Coverage (%)
    # -----------------------------------------
    threshold = 0.5                       # Heatmap threshold for "mold"
    mold_pixels = np.sum(cam > threshold)
    total_pixels = cam.size
    mold_percent = (mold_pixels / total_pixels) * 100

    print(f"\nEstimated Mold Coverage: {mold_percent:.2f}%")

    # -----------------------------------------
    # 6. Save Heatmap Visualization
    # -----------------------------------------
    original = cv2.imread(img_path)       # Load original image (for overlay)
    cam_resized = cv2.resize(cam, (original.shape[1], original.shape[0]))

    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original
    result = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Save final output
    cv2.imwrite("gradcam_result.jpg", result)
    print("Grad-CAM saved as gradcam_result.jpg")

# ============================================================
# MAIN SCRIPT ENTRY
# ============================================================

if len(sys.argv) < 2:
    print("Usage: python3 gradcam_viewer.py <image.jpg>")
    sys.exit()

img_path = sys.argv[1]

print("Loading model...")
model = load_model("fruit_freshness_cnn.h5", compile=False)

gradcam_with_mold_detection(model, img_path)

