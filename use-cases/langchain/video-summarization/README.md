# Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-v-2_6  

## Installation
* [Install Intel client GPU](https://dgpu-docs.intel.com/driver/client/overview.html)
* [Install OpenVino Runtime from archive](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)
* Install openvino-genai and set up python enviornment. Currently must be root user.

```
conda create -n ovlangvidsumm python=3.10
conda activate ovlangvidsumm
pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git
pip install nncf openvino-genai timm einops langchain decord
```

## Convert and Save MiniCPM-V-2_6
```
optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 --trust-remote-code --weight-format int4 MiniCPM_INT4 # int8 also available 
```

## Run Video Summarization Examples
```
bash driver.sh
```