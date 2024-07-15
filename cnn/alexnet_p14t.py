import cv2
import os
import numpy as np
import pandas as pd
import pickle
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from alexnet_pytorch import AlexNet
import argparse
import gzip
import torch
import torch.nn as nn
import gc

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=1024):  # Assuming you want 1024 features
        super(CustomAlexNet, self).__init__()
        self.alexnet = AlexNet.from_pretrained('alexnet')
        
        # Modify the classifier to output 1024 features
        self.alexnet.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # First layer from original AlexNet
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # Second layer from original AlexNet
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # New layer to output 1024 features
        )

    def forward(self, x):
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier(x)
        return x




class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file, sep='|')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['video'].replace('/1', ''))
        print(video_path)
        images = sorted(glob.glob(video_path))

        frames = []
        for image_path in images:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        sample = {
            'frames': frames,  # List of transformed frames
            'gloss': self.data_frame.iloc[idx]['orth'],
            'text': self.data_frame.iloc[idx]['translation'],
            'name': self.data_frame.iloc[idx]['name'],
            'signer': self.data_frame.iloc[idx]['speaker']
        }

        return sample


def extract_features_per_frame(model, video_frames, device):
    model.eval()
    features_list = []
    with torch.no_grad():
        for frame in video_frames:
            frame = frame.to(device)
            features = model(frame.unsqueeze(0))  # Process each frame through the model
            features_list.append(features.squeeze(0).cpu())  # Remove batch dimension and move to CPU
    return features_list


def main():
    parser = argparse.ArgumentParser(description='Extract features using AlexNet.')
    parser.add_argument('--data', type=str, required=True, help='Specify the dataset to use')
    parser.add_argument('--size', type=int, default=1024, help='Specify the size of the feature vector')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (should be 1 for variable-length videos)')

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(csv_file='./dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{}.corpus.csv'.format(args.data), root_dir='./dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{}/'.format(args.data), transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CustomAlexNet(num_classes=args.size).to(device)
    all_data = []
    for i, sample in enumerate(dataset):
        print(f"Sample {i}: {sample.keys()}")
        print(f"Number of frames: {len(sample['frames'])}")
    
        # Process frames of the current sample
        video_frames = sample['frames']  # Directly use the 'frames' from the sample
        video_features = []
        
        # Assuming extract_features_per_frame is designed to process a single frame
        for frame in video_frames:
            frame = frame.to(device)
            features = model(frame.unsqueeze(0))  # Add batch dimension
            video_features.append(features.squeeze(0).cpu())  # Remove batch dimension and move to CPU
        
        # Stack all features to get a tensor for the whole video
        video_features_tensor = torch.stack(video_features)
        
        print(f"Processed Sample {i}: SHAPE: {video_features_tensor.shape}")
        
        # Collect and store additional fields from the sample
        all_data.append({
            'sign': video_features_tensor,
            'gloss': sample['gloss'],
            'text': sample['text'],
            'name': sample['name'],
            'signer': sample['signer']
        })

        del video_frames, video_features  # CHANGE: Explicitly delete variables to free memory
        gc.collect()  # CHANGE: Collect garbage to free memory
        torch.cuda.empty_cache()  # CHANGE: Empty CUDA cache to free unused memory
    
    # Save the processed features and additional information to a file
    pickle_file = f'alexnet_features_{args.data}.gz'
    with gzip.open(pickle_file, 'wb') as f:
        pickle.dump(all_data, f)

    gc.collect()
    print(f"Data saved to {pickle_file}")

if __name__ == "__main__":
    main()