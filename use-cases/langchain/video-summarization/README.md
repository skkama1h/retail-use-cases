# Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6  

## Installation
* Install Intel Client GPU and OpenVINO Runtime from Archive.
'''
# Validated on Ubuntu 24.04 and 22.04
bash install.sh
'''

* Create Conda Enviornment and Set Up Python Enviornment 
```
conda create -n ovlangvidsumm python=3.10
conda activate ovlangvidsumm
pip install -r requirements.txt
```

## Convert and Save Optimized MiniCPM-V-2_6
```
optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 --trust-remote-code --weight-format int8 MiniCPM_INT8 # int4 also available 
```

## Run Video Summarization
```
python video_summarizer.py /path/to/your_video.mp4 MiniCPM_INT8/ -d "GPU" -r 480 270 
```