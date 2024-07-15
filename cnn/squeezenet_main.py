import cv2
import os
import numpy as np
import pandas as pd
import csv
import glob
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.mobilenet import preprocess_input

from squeezenet import SqueezeNet
from squeezenet import SqueezeNet_11
from squeezenet import output
from squeezenet import create_fire_module
from squeezenet import get_axis

def preprocess_frame(frame):
    frame = cv2.resize(frame, (227, 227))  # Resize to 227x227 for SqueezeNet
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)  # Preprocess for SqueezeNet, ensure compatibility
    return frame

def extract_and_aggregate_features(model, image_paths):
    features = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        processed_frame = preprocess_frame(frame)
        feature = model.predict(processed_frame)
        features.append(feature)
    aggregated_features = np.mean(np.array(features), axis=0)
    return aggregated_features

squeezenet_base = SqueezeNet(input_shape=(227, 227, 3), nb_classes=1000)

# Add new layers
x = Dense(2560, activation='relu')(squeezenet_base.output)

# Create the custom model
squeezenet_model = Model(inputs=squeezenet_base.input, outputs=x)

base_dir = './PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev/'

# Load the CSV file and split the data
csv_file_path = './PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv'
data = pd.read_csv(csv_file_path, sep='|')
split_data = data

# Extract relevant columns
image_sequence_paths_temp = split_data['video'].tolist()
sequence_glosses = split_data['orth'].tolist()
text_translations = split_data['translation'].tolist()

image_sequence_paths = [path.replace('/1', '') for path in image_sequence_paths_temp]

data_to_save = []
for i, img_path in enumerate(image_sequence_paths):
    full_path = os.path.join(base_dir, img_path)

    images = glob.glob(full_path)
    if images:
        aggregated_features = extract_and_aggregate_features(squeezenet_model, images)
        features_string = ','.join([f'{num:.17g}' for num in aggregated_features.flatten()])  # Join all features into a single string
        data_to_save.append({
            "features": features_string,  # Now features are a single string
            "gloss": sequence_glosses[i],
            "translation": text_translations[i]
        })
    else:
        print(f"No images found in {full_path}.")


# Convert to DataFrame and save to CSV
df = pd.DataFrame(data_to_save)
csv_file = 'train_aggregated_features.csv'
df.to_csv(csv_file, index=False, float_format='%.17g', quoting=csv.QUOTE_NONE, escapechar='\\')
print(f"Data saved to {csv_file}")

cv2.destroyAllWindows()
