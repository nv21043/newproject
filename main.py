import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Register custom objects for model loading
tf.keras.utils.get_custom_objects().update({
    'Sequential': Sequential,
    'Conv2D': Conv2D,
    'MaxPooling2D': MaxPooling2D,
    'Flatten': Flatten,
    'Dense': Dense
})

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Correct path to model.json
model_json_path = 'C:/Users/sa987/Downloads/new-project/models/model.json'
model_weights_path = 'C:/Users/sa987/Downloads/new-project/keras_Model.h5'
labels_path = 'C:/Users/sa987/Downloads/new-project/labels.txt'

# Load the model configuration
try:
    with open(model_json_path, 'r') as f:
        model_config = f.read()
except FileNotFoundError:
    print(f"File not found: {model_json_path}")
    raise

# Load the model from JSON configuration
try:
    model = tf.keras.models.model_from_json(model_config)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load the weights
try:
    model.load_weights(model_weights_path)
except FileNotFoundError:
    print(f"File not found: {model_weights_path}")
    raise

# Load the labels
try:
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"File not found: {labels_path}")
    raise

# Image preprocessing functions
def preprocess_image(image, target_size=(224, 224)):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def pad_image(image, target_size=(224, 224)):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(target_size[1]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))
    padded_image = np.full((target_size[1], target_size[0], 3), 0, dtype=np.uint8)
    padded_image[
        (target_size[1] - new_size[0]) // 2 : (target_size[1] + new_size[0]) // 2,
        (target_size[0] - new_size[1]) // 2 : (target_size[0] + new_size[1]) // 2,
    ] = image
    return padded_image

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Grab the web camera's image.
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image")
        break

    # Choose to resize or pad based on the image dimensions
    if image.shape[0] > 224 or image.shape[1] > 224:
        processed_image = preprocess_image(image, target_size=(224, 224))
    else:
        processed_image = pad_image(image, target_size=(224, 224))

    # Show the processed image in a window
    cv2.imshow("Webcam Image", processed_image)

    # Make the image a numpy array and reshape it to the model's input shape.
    processed_image = np.asarray(processed_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    processed_image = (processed_image / 127.5) - 1

    # Predict using the model
    try:
        prediction = model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name, end="")
        print(" Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    except Exception as e:
        print(f"Error during prediction: {e}")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # 27 is the ASCII for the ESC key on your keyboard.
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
