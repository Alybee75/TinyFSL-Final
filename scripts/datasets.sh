#!/bin/bash

# Create directory for all datasets
mkdir -p datasets
cd datasets

# Create directory for FSL datasets and download the datasets
mkdir -p fsl-datasets
# Assuming gdown is installed. If not, you can install it using `pip install gdown`
# Use gdown to download the files from Google Drive. Replace the ID with the actual file IDs.
# For demonstration, only individual files are handled since folders cannot be directly downloaded.

# Download FSL datasets
gdown --folder https://drive.google.com/drive/folders/13qM-WamOtbZZYbffF1Pcru7nxoIU9Zmk -O ./fsl-datasets

# Create and navigate to cnn-phoenix directory
mkdir -p cnn-phoenix
cd cnn-phoenix

# Download mobilenet
gdown --folder https://drive.google.com/drive/folders/13GJP66qcK-Ucrb3Z-v27AguQtgQ8HhHG -O ./mobilenet

# Download iv4
gdown --folder https://drive.google.com/drive/folders/1G8P9cmOHTaksr7qCsc6pwNjYXtZcMXc6 -O ./iv4

# Download squeezenet
gdown --folder https://drive.google.com/drive/folders/1dv8E03aQ5YY7SwuvDCJe7AQSyHXIFPmx -O ./squeezenet

# Download alexnet
gdown --folder https://drive.google.com/drive/folders/1agTPpjuKCQGyROhMX9XXnVyG8JpHYqrj -O ./alexnet

# Download camgoz files
mkdir -p camgoz
cd camgoz

wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.train"
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.dev"
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.test"