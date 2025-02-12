# Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6

## Installation
* Install Intel Client GPU, Conda, and Set Up Python Enviornment
```
# Validated on Ubuntu 24.04 and 22.04
bash install.sh
```

## Convert and Save Optimized MiniCPM-V-2_6
First, follow the steps on the [MiniCPM-V-2_6 HuggingFace Page](https://huggingface.co/openbmb/MiniCPM-V-2_6) to gain access to the model. For more information on user access tokens for access to gated models see [here](https://huggingface.co/docs/hub/en/security-tokens).

Next, convert and save the optimized model.
```
optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 --trust-remote-code --weight-format int8 MiniCPM_INT8 # int4 also available 
```

## Run Video Summarization
Summarize a video using `video_summarizer.py`. For example, start by downloading the following video:
 
```
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
```

Now create a summary of that video using `video_summarizer.py`:
```
python video_summarizer.py one-by-one-person-detection.mp4 MiniCPM_INT8/ -d "GPU" -r 480 270 
```