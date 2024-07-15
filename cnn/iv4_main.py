import cv2
import os
import gc
import numpy as np
import pandas as pd
import gzip
import csv
import glob
import pickle
import torch
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers import Dense
from keras.applications.mobilenet import preprocess_input
import tensorflow as tf

from keras.layers import Add, Activation, Concatenate, Dropout
from keras.layers import Flatten, MaxPooling2D
import keras.backend as K
from inception_v4 import create_model

import argparse

def preprocess_frame(frame):
    frame = cv2.resize(frame, (299, 299))  # Resize to 299x299 for InceptionV4
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)  # Preprocess for InceptionV4
    return frame
    
def extract_features(model, image_paths):
    features = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        processed_frame = preprocess_frame(frame)
        feature = model.predict(processed_frame)
        features.append(feature.squeeze())  # Remove single-dimensional entries
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    return torch.from_numpy(features_tensor.numpy())
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, help='Specify the dataset to use (e.g., train, dev, test)')
    parser.add_argument('--size', type=int, help='Specify the size of something (e.g., 1024)')

    args = parser.parse_args()

    # Now you can use args.data and args.size in your script
    print(f"Data argument received: {args.data}")
    print(f"Size argument received: {args.size}")

    # Here you would add the rest of your script logic,
    # using args.data and args.size where necessary
    base_dir = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{}/'.format(args.data)
    # Load the CSV file and split the data
    csv_file_path = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{}.corpus.csv'.format(args.data)  # Update with the actual path
    data = pd.read_csv(csv_file_path, sep='|')  # Use the correct separator
    split_data = data
    data_to_save = []
    # Extracting the relevant columns
    image_sequence_paths_temp = split_data['video'].tolist()
    names = split_data['name'].tolist()
    speakers = split_data['speaker'].tolist()
    sequence_glosses = split_data['orth'].tolist()
    text_translations = split_data['translation'].tolist()
    prefix = "./dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{}/".format(args.data)
    image_sequence_paths = [path.replace('/1', '') for path in image_sequence_paths_temp]

    base_model = create_model(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(args.size, activation='relu')(x)
    custom_model = Model(inputs=base_model.input, outputs=x)


    for i,img_path in enumerate(image_sequence_paths):
        full_path = os.path.join(base_dir, img_path)
                                                                        

        images = glob.glob(full_path)
        if images:
            ext_features = extract_features(custom_model, images)

        
            data_to_save.append({
                "sign": ext_features,  # Store features as a string
                "gloss": sequence_glosses[i],
                "text": text_translations[i],
                "name": names[i],
                "signer": speakers[i]
            })
            
        else:
            print(f"No images found in {images}.")



    # Save to CSV
    pickle_file = 'phoenix.inceptionv4.{}.gz'.format(args.data)
    with gzip.open(pickle_file, 'wb') as f:
        pickle.dump(data_to_save, f)

    gc.collect()
    print(f"Data saved to {pickle_file}")

if __name__ == "__main__":
    main()


    




