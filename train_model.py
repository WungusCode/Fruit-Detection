"""
============================================================
CPSC 483 – Fruit Freshness Detection Project
TRAINING SCRIPT (FULLY COMMENTED)
------------------------------------------------------------
This script trains a Convolutional Neural Network (CNN)
from scratch to classify fruit images as fresh or rotten.
It also evaluates the model and saves it for Grad-CAM use.
============================================================
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# ============================================================
# DATASET CONFIGURATION
# ============================================================

train_path = "dataset/train"     # Folder containing training images
test_path  = "dataset/test"      # Folder containing testing images

img_size = 150                   # All images will be resized to 150×150
batch_size = 32                 # Number of images processed per batch
epochs = 15                     # Number of training cycles

# ============================================================
# DATA PREPROCESSING + DATA AUGMENTATION
# ============================================================

# Training data generator with augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values (0–1)
    rotation_range=20,          # Random rotations
    zoom_range=0.2,             # Random zoom-in/out
    horizontal_flip=True        # Random horizontal flips
)

# Testing data has NO augmentation (only normalization)
test_gen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"         # Two classes: fresh or rotten
)

# Load testing dataset
test_data = test_gen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False               # Keep order for evaluation
)

# ============================================================
# BUILD CNN MODEL (FROM SCRATCH)
# ============================================================

# Input layer: receives a 150×150 RGB image
inputs = Input(shape=(img_size, img_size, 3))

# Convolution Block 1: learns simple edges and color gradients
x = Conv2D(32, (3,3), activation='relu', name="conv1")(inputs)
x = MaxPooling2D(2,2)(x)        # Reduces spatial size

# Convolution Block 2: learns textures and surface irregularities
x = Conv2D(64, (3,3), activation='relu', name="conv2")(x)
x = MaxPooling2D(2,2)(x)

# Convolution Block 3: learns complex mold patterns
x = Conv2D(128, (3,3), activation='relu', name="conv3")(x)
x = MaxPooling2D(2,2)(x)

# Flatten converts feature maps → 1D vector
x = Flatten()(x)

# Dense layer: makes final decision based on features
x = Dense(128, activation='relu')(x)

# Dropout: prevents overfitting by disabling 30% of neurons
x = Dropout(0.3)(x)

# Output layer: probability of being Rotten (1) or Fresh (0)
outputs = Dense(1, activation='sigmoid')(x)

# Build the model object
model = Model(inputs, outputs)

# Compile with optimizer, loss function, and metrics
model.compile(
    optimizer=Adam(0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Print model structure for debugging
model.summary()

# ============================================================
# TRAIN THE MODEL
# ============================================================

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data   # Used to detect overfitting
)

# ============================================================
# PLOT TRAINING RESULTS
# ============================================================

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Over Time")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Time")
plt.show()

# ============================================================
# EVALUATION ON TEST DATASET
# ============================================================

pred = model.predict(test_data)                       # Get model predictions
predicted_labels = (pred > 0.5).astype(int).reshape(-1)

print("\nClassification Report:")
print(classification_report(test_data.classes, predicted_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(test_data.classes, predicted_labels))

# ============================================================
# SAVE THE MODEL
# ============================================================

model.save("fruit_freshness_cnn.h5")   # Saves trained model for later use
print("\nModel saved as fruit_freshness_cnn.h5")

