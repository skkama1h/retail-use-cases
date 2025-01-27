# Generate Video Summarizations Using Langchain and MiniCPM-v-2_6  

## Installation
* [Install Intel client GPU](https://dgpu-docs.intel.com/driver/client/overview.html)
* [Install OpenVino Runtime from archive](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)
* Install openvino-genai and set up python enviornment

```
conda create -n ovlangvidsumm python=3.10
conda activate ovlangvidsumm
pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git
pip install nncf openvino-genai timm einopsaa
pip install -r reqs.txt
```

## Run Examples
```
bash driver.sh
```