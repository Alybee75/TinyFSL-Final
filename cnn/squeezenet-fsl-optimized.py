import cv2
import os
import gc
import numpy as np
import pandas as pd
import gzip
import pickle
import glob
import tensorflow as tf
import argparse
import torch

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.mobilenet import preprocess_input
from squeezenet import SqueezeNet
from squeezenet import SqueezeNet_11
from squeezenet import output
from squeezenet import create_fire_module
from squeezenet import get_axis

def preprocess_frame(frame):
    # Resize and preprocess for SqueezeNet
    frame = cv2.resize(frame, (227, 227))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

def extract_features_from_video(model, video_path, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    features = []
    batch_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if len(batch_frames) > 0:
                batch_features = model.predict(np.array(batch_frames))
                features.extend(batch_features.squeeze())
            break

        processed_frame = preprocess_frame(frame)
        batch_frames.append(processed_frame.squeeze())

        if len(batch_frames) == batch_size:
            batch_features = model.predict(np.array(batch_frames))
            features.extend(batch_features.squeeze())
            batch_frames = []
            print(f"Processed a batch of {batch_size} frames")

    cap.release()
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    print("Total features extracted:", len(features))
    return torch.from_numpy(features_tensor.numpy())

def main():
    parser = argparse.ArgumentParser(description='Feature extraction with SqueezeNet')
    parser.add_argument('--data', type=str, help='Specify the dataset to use')
    parser.add_argument('--size', type=int, help='Specify the size of the output layer')

    args = parser.parse_args()

    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print(f"Num GPUs Available: {num_gpus}")
    if num_gpus > 0:
        print("Using GPU for processing")
    else:
        print("No GPU found, using CPU")

    base_dir = './dataset/FSL/FSL-105/data/'
    csv_file_path = './dataset/FSL/FSL-105/fsl-csv/{}.csv'.format(args.data)
    data = pd.read_csv(csv_file_path, sep=',')

    image_sequence_paths_temp = data['vid_path'].tolist()
    sequence_glosses = data['gloss'].tolist()
    text_translations = data['label'].tolist()
    
    # Initialize SqueezeNet model with a custom top layer
    base_model = SqueezeNet(input_shape=(227, 227, 3), nb_classes=1000)
    x = base_model.output
    x = Dense(args.size, activation='relu')(x) if args.size else x  # Conditional dense layer based on args.size
    custom_model = Model(inputs=base_model.input, outputs=x)

    data_to_save = []
    for i, img_path in enumerate(image_sequence_paths_temp):
        full_path = os.path.join(base_dir, img_path.replace("/1", ""))
        full_path = full_path.replace("\\", "/")
        print("TRYING PATH:", full_path)

        ext_features = extract_features_from_video(custom_model, full_path)

        data_to_save.append({
            "sign": ext_features,  # Store features as a tensor
            "gloss": sequence_glosses[i],
            "text": text_translations[i],
            "name": image_sequence_paths_temp[i],
            "signer": "signer01"
        })

    # Save to GZIP pickle
    pickle_file = 'fsl.squeezenet.{}.gz'.format(args.data)
    with gzip.open(pickle_file, 'wb') as f:
        pickle.dump(data_to_save, f)

    gc.collect()
    print(f"Data saved to {pickle_file}")

if __name__ == "__main__":
    main()
