#!/bin/bash
# Download FER2013 from Kaggle
# Requires: kaggle.json in ~/.kaggle/
cd "$(dirname "$0")/.."
mkdir -p data
cd data
kaggle datasets download -d msambare/fer2013
unzip -q fer2013.zip
rm fer2013.zip
ls -lh fer2013.csv
echo "✓ FER2013 downloaded to data/"
