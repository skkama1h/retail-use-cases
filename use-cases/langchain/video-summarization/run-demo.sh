#!/bin/bash

source activate-conda.sh
activate_conda
conda activate ovlangvidsumm

if [ "$1" == "--skip" ]; then
	echo "Skipping sample video download"
else
    # Download sample video
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
fi

echo "Run Video Summarization"
INPUT_FILE="one-by-one-person-detection.mp4"
DEVICE="GPU"
RESOLUTION_X=480
RESOLUTION_Y=270
python video_summarizer.py INPUT_FILE MiniCPM_INT8/ -d "GPU" -r RESOLUTION_X RESOLUTION_Y
