from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from keras.applications.inception_v4 import preprocess_input
from keras.applications.inception_v4 import InceptionV4
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def preprocess_frame(frame):
    # Resize frame to 299x299 for InceptionV4
    frame = cv2.resize(frame, (299, 299))
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
# model is your InceptionV4 model
video_paths = [...]

base_model = InceptionV4(weights='imagenet', include_top=False)

# Add global average pooling layer
x = GlobalAveragePooling2D()(base_model.output)
# Add a fully connected layer with 2560 units
x = Dense(2560, activation='relu')(x)
index=0
video_features=[]
for video_path in video_paths:
    video_features[index] = extract_features(base_model,video_path)
    index+=1
# Create the new model
