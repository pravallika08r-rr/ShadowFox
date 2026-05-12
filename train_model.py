import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=32
)

# Load testing dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32
)

# Get class names
class_names = train_dataset.class_names
print("Classes:", class_names)

# Normalize images
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# Build CNN model
model = models.Sequential([

    # First Convolution Layer
    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    ),

    layers.MaxPooling2D(),

    # Second Convolution Layer
    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D(),

    # Third Convolution Layer
    layers.Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D(),

    # Flatten Layer
    layers.Flatten(),

    # Dense Layer
    layers.Dense(
        128,
        activation='relu'
    ),

    # Dropout Layer
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(
        len(class_names),
        activation='softmax'
    )
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=15
)

# Save trained model
model.save("image_classifier_model.keras")

print("Model trained and saved successfully!")

# Plot Training Accuracy
plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

# Plot Validation Accuracy
plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

# Labels
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Legend
plt.legend()

# Title
plt.title("Training vs Validation Accuracy")

# Show graph
plt.show()