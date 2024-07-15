import cv2
import os
import gc
import numpy as np
import pandas as pd
import glob
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.layers import Reshape, Conv2D, UpSampling2D
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from squeezenet import SqueezeNet_base
from keras_tuner import HyperModel, RandomSearch
import argparse
from tensorflow.keras.models import load_model
from keras.backend import clear_session
import pickle
import gzip
import ast

import torch

class VideoModelHyperTuner(HyperModel):
    def __init__(self, base_model,labels_train):
        self.base_model = base_model
        self.labels_train = labels_train
    def build(self, hp):
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        dense_units = hp.Int('dense_units', min_value=32, max_value=1024, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        num_filters = hp.Int('num_filters', min_value=32, max_value=512, step=32)

        x = GlobalAveragePooling2D()(self.base_model.output)
        x = Dense(units=dense_units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        # Reshape to add spatial dimensions (1x1)
        x = Reshape((1, 1, dense_units))(x)

        # Upsample to increase spatial dimensions to (165, 1)
        x = UpSampling2D(size=(self.labels_train, 1))(x)

        # Convolution to increase width from 1 to 1024
        x = Conv2D(filters=num_filters, kernel_size=(1, 32), strides=(1, 32), padding='same', activation='relu')(x)

        # Final convolution to adjust the depth to 1024
        x = Conv2D(1024, (1, 1), padding='same', activation='sigmoid')(x)

        # Flatten the second dimension
        x = Reshape((165, 1024))(x)  # Reshape to match the target shape (None, 165, 1024)

        model = Model(inputs=self.base_model.output, outputs=x)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

        print("MODEL OUTPUT SHAPE", model.output_shape)
        return model

def preprocess_frame(frame):
    frame = cv2.resize(frame, (227, 227))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = tf.keras.applications.mobilenet.preprocess_input(frame)
    return frame
def pad_features(features, target_shape):
    padded_features = []
    for feature in features:
        if feature.shape != target_shape:
            padded_feature = np.zeros(target_shape)
            slices = tuple(slice(0, min(dim_size, target_dim)) for dim_size, target_dim in zip(feature.shape, target_shape))
            padded_feature[slices] = feature[slices]
            padded_features.append(padded_feature)
        else:
            padded_features.append(feature)
    return np.array(padded_features)
def find_max_dimensions(labels_list):
    # Find the maximum number of rows and columns across all label sets
    max_dim1 = max(len(label) for labels in labels_list for label in labels)  # max number of rows
    max_dim2 = max(max(len(item) for item in label) for labels in labels_list for label in labels)  # max length of inner lists
    return max_dim1, max_dim2
def standardize_labels(labels, max_dim1, max_dim2, pad_value=0.0):
    # Standardize labels according to the maximum dimensions provided
    standardized_labels = []
    for label in labels:
        # Pad each inner list to max_dim2
        padded_label = [item + [pad_value] * (max_dim2 - len(item)) for item in label]
        # If the outer list is shorter than max_dim1, pad it with lists of zeros
        padded_label += [[pad_value] * max_dim2] * (max_dim1 - len(label))
        standardized_labels.append(padded_label)

    return np.array(standardized_labels, dtype=np.float32)
def extract_features(model, image_paths):
    features = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        processed_frame = preprocess_frame(frame)
        feature = model.predict(processed_frame)
        features.append(feature.squeeze())
    if features:
        aggregated_features = np.mean(features, axis=0)
        print("TRAIN SHAPE:", aggregated_features.shape)
        return aggregated_features
    else:
        return np.array([])
def extract_features_trained(base_model, trained_model, image_paths):
    features = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        processed_frame = preprocess_frame(frame)
        feature_base = base_model.predict(processed_frame)
        feature = trained_model.predict(feature_base)
        features.append(feature.squeeze())  # Remove single-dimensional entries
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    return torch.from_numpy(features_tensor.numpy())
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, help='Specify the dataset to use (e.g., train, dev, test)')
    parser.add_argument('--size', type=int, help='Specify the size of something (e.g., 1024)')
    parser.add_argument('--retrain', type=str, help='Specify whether or not to train on Teacher model (true or false)')
    parser.add_argument('--trained', type=str, help='Specify whether or not to run on trained h5 (true or false)')
    args = parser.parse_args()

    # Now you can use args.data and args.size in your script
    print(f"Data argument received: {args.data}")
    print(f"Size argument received: {args.size}")
    #dataset = pd.read_csv('../TinyFSL/notes/train_check_train.csv')
    dataset_val = pd.read_csv('../TinyFSL/notes/train_check_val2.csv')

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
    image_sequence_paths = [path.replace('/1', '') for path in image_sequence_paths_temp]
    train_dir = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/'
    base_dir = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{}/'.format(args.data)
    
    base_model = SqueezeNet_base(input_shape=(227, 227, 3))
    print("--------------------------------BASE MODEL OUTPUT:", base_model.output)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(args.size, activation='relu')(x)
    custom_model = Model(inputs=base_model.input, outputs=x)

    if args.retrain == "true":
        features_val = []
        labels_val = []
        for _, row in dataset_val.iloc[:2].iterrows():
            video_path = os.path.join(train_dir, row['name'])
            video_path = video_path + "/*.png"
            images = glob.glob(video_path)
            features = extract_features(base_model, images)
            labels = ast.literal_eval(row['sign'])
            features_val.append(features)
            labels_val.append(labels)
        
        
        # Extract features and labels for the next 5 rows for the training dataset
        features_train = []
        labels_train = []
        for _, row in dataset_val.iloc[2:7].iterrows():
            video_path = os.path.join(train_dir, row['name'])
            video_path = video_path + "/*.png"
            images = glob.glob(video_path)
            features = extract_features(base_model, images)
            features_train.append(features)  # append the aggregated features
            labels = ast.literal_eval(row['sign'])  # Ensure labels are in the correct shape
            labels_train.append(labels)
        max_dim1, max_dim2 = find_max_dimensions([labels_train, labels_val])    
        labels_val = standardize_labels(labels_val, max_dim1, max_dim2)        
        labels_val = np.array(labels_val).astype('float32')
        print("LABELS VAL SHAPE:", labels_val.shape)
        labels_train = standardize_labels(labels_train, max_dim1, max_dim2)
        labels_train = np.array(labels_train).astype('float32') 
        print("LABELS TRAIN SHAPE:", labels_train.shape)
        print("----------------------------------------------------DONE APPENDING DATA!----------------------------------------------------")
        features_val = np.stack(features_val) if len(features_val) > 0 else np.array([])
        features_train = np.stack(features_train) if len(features_train) > 0 else np.array([])
        tuner = RandomSearch(
            VideoModelHyperTuner(base_model, labels_train.shape[1]),
            objective='val_loss',
            max_trials=10,
            executions_per_trial=2,
            directory='model_tuning',
            project_name='VideoFeatureExtraction'
        )

        print("----------------------------------------------------DONE SETTING UP TUNER!----------------------------------------------------")
        tuner.search(x=features_train, y=labels_train, epochs=10, validation_data=(features_val, labels_val))
        print("----------------------------------------------------DONE TUNING!----------------------------------------------------")
        custom_model = tuner.get_best_models(num_models=1)[0]
        custom_model.save('./best_custom_iv4.h5')
        print("----------------------------------------------------DONE SAVING MODEL!----------------------------------------------------")

    if args.trained == "true":
        custom_model = load_model('./best_custom_iv4.h5')

    for i, img_path in enumerate(image_sequence_paths):
        full_path = os.path.join(base_dir, img_path)
        if full_path.endswith('/*.png'):
            full_path = full_path[:-6] 

        if os.path.isdir(full_path) and len(glob.glob(os.path.join(full_path, '*.png'))) > 0:
            ext_features = extract_features_trained(base_model, custom_model, glob.glob(full_path + '/*.png'))
            data_to_save.append({
                "sign": ext_features.numpy().tolist(),
                "gloss": sequence_glosses[i],
                "text": text_translations[i],
                "name": names[i],
                "signer": speakers[i]
            })
        else:
            print(f"No images found in {full_path}.")

    # Save to a pickle file
    with gzip.open('fsl.inceptionv4.{}.gz'.format(args.data), 'wb') as f:
        pickle.dump(data_to_save, f)

    gc.collect()
    print(f"Data saved to fsl.inceptionv4.{args.data}.gz")

if __name__ == "__main__":
    main()
