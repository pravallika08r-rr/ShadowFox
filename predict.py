import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model(
    "image_classifier_model.keras"
)

# Class names
class_names = ['car', 'cat', 'dog']

# Image path
img_path = "sample.jpg"

# Load image
img = image.load_img(
    img_path,
    target_size=(128, 128)
)

# Convert image to array
img_array = image.img_to_array(img)

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Normalize image
img_array = img_array / 255.0

# Predict image
prediction = model.predict(img_array)

# Predicted class
predicted_class = class_names[np.argmax(prediction)]

# Confidence score
confidence = np.max(prediction) * 100

# Print results
print("Predicted Class:", predicted_class)
print("Confidence:", confidence, "%")