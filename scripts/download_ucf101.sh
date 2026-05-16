#!/bin/bash
set -e
mkdir -p /shared/ssd/home/b-s-adhikari/nn-gpt/VJEPA2/data/ucf101
cd /shared/ssd/home/b-s-adhikari/nn-gpt/VJEPA2/data/ucf101
echo "Downloading UCF101..."
wget -q -c --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
echo "Extracting UCF101..."
7z x -y UCF101.rar > /dev/null
echo "Downloading splits..."
wget -q -c --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip -o UCF101TrainTestSplits-RecognitionTask.zip > /dev/null
echo "Creating CSV..."
# Create a CSV of absolute path -> label (0-100)
ls UCF-101/*/*.avi | awk -F'/' '{print $2}' | sort | uniq | awk '{print $1" "NR-1}' > class_mapping.txt
rm -f ucf101_train.csv
while read -r class label; do
    find "$PWD/UCF-101/$class" -name "*.avi" | awk -v lbl="$label" '{print $1" "lbl}' >> ucf101_train.csv
done < class_mapping.txt
echo "Done! Wrote $(wc -l < ucf101_train.csv) videos to ucf101_train.csv"
