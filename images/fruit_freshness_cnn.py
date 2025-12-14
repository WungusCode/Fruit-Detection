import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os

# ============================================================
# 1. Dataset paths
# ============================================================
train_path = "dataset/train"
test_path  = "dataset/test"

img_size = 150
batch_size = 32

print("CURRENT DIRECTORY:", os.getcwd())
print("FILES:", os.listdir())

# ============================================================
# 2. Image preprocessing
# ============================================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path, target_size=(img_size, img_size),
    batch_size=batch_size, class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    test_path, target_size=(img_size, img_size),
    batch_size=batch_size, class_mode="binary", shuffle=False
)

# ============================================================
# 3. Functional CNN model (fixed for Grad-CAM)
# ============================================================
inputs = Input(shape=(img_size, img_size, 3))

x = Conv2D(32, (3,3), activation='relu', name="conv1")(inputs)
x = MaxPooling2D(2,2)(x)

x = Conv2D(64, (3,3), activation='relu', name="conv2")(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(128, (3,3), activation='relu', name="conv3")(x)
x = MaxPooling2D(2,2)(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(0.0005), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# ============================================================
# 4. Train
# ============================================================
history = model.fit(train_data, epochs=3, validation_data=test_data)

# ============================================================
# 5. Graphs
# ============================================================
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.legend(); plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend(); plt.title("Loss")
plt.show()

# ============================================================
# 6. Evaluation
# ============================================================
pred = model.predict(test_data)
predicted_labels = (pred > 0.5).astype(int).reshape(-1)

print("\nClassification Report:")
print(classification_report(test_data.classes, predicted_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(test_data.classes, predicted_labels))

model.save("fruit_freshness_cnn.h5")
print("\nModel saved.")

# ============================================================
# 7. Grad-CAM (RELIABLE VERSION)
# ============================================================
def gradcam(model, img_path, layer_name="conv3"):
    if not os.path.exists(img_path):
        print("ERROR: File not found:", img_path)
        print("Directory contains:", os.listdir())
        return

    print("\nRunning Grad-CAM on:", img_path)

    # Load + preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Build Grad-CAM model
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        pred_score = preds[0]

    # Gradients
    grads = tape.gradient(pred_score, conv_output)[0]   # tensor -> numpy automatically later
    conv_output = conv_output[0]                        # tensor

    # Compute channel weights
    weights = tf.reduce_mean(grads, axis=(0, 1))        # shape = (channels,)

    # Build CAM manually (no .numpy(), works on all TF versions)
    cam = np.zeros(conv_output.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    # Normalize CAM
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()

    # Load original image
    original = cv2.imread(img_path)

    # Resize CAM to original image size
    cam = cv2.resize(cam, (original.shape[1], original.shape[0]))

    # Convert CAM to heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original image
    output = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("gradcam_result.jpg", output)
    print("Grad-CAM saved as: gradcam_result.jpg")

# ============================================================
# 8. RUN GRAD-CAM
# ============================================================
TEST_IMAGE = "bananas.jpg"

gradcam(model, TEST_IMAGE)

