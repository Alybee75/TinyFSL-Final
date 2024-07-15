from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV3Small  # or MobileNetV3Large, depending on your needs
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def preprocess_frame(frame):
    # Resize frame to 224x224 for MobileNetV3
    frame = cv2.resize(frame, (224, 224))
    # Convert frame to array and add batch dimension
    frame = np.expand_dims(frame, axis=0)
    # Normalize frame
    frame = preprocess_input(frame)
    return frame

def extract_features(model, video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame)
        feature = model.predict(processed_frame)
        features.append(feature)
    cap.release()
    return np.array(features)

# Example usage
video_paths = [...]  # Your video paths

base_model = MobileNetV3Small(weights='imagenet', include_top=False)  # or MobileNetV3Large

# Add global average pooling layer
x = GlobalAveragePooling2D()(base_model.output)
# Add a fully connected layer (adjust units as needed)
x = Dense(1024, activation='relu')(x)  # Adjusted number of units

# Create a new model
model = Model(inputs=base_model.input, outputs=x)

# Extract features for each video
video_features = []
for video_path in video_paths:
    features = extract_features(model, video_path)
    video_features.append(features)

# video_features now contains the features extracted from each video
