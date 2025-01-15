# QnA
Demonstrates a pipeline which performs QnA. The primary components utilize OpenVINO™ in LangChain for audio-speech-recognition, LLM text generation/response, and text-to-speech (currently [OuteAI/OuteTTS-0.1-350M](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/outetts-text-to-speech/outetts-text-to-speech.ipynb)).

## Installation

Get started by running the below command.

```
./install.sh
```

Note: if this script has already been performed and you'd like to install code change only then the below command can be used instead to skip the re-install of dependencies.

```
./install --skip
```

## Run Examples

This sample requires an audio file. A sample wav file can be downloaded [here](https://github.com/intel/intel-extension-for-transformers/raw/refs/heads/main/intel_extension_for_transformers/neural_chat/assets/audio/sample_2.wav)

Run the below command to start the demo with the following defaults:

LLM Model: llmware/llama-3.2-3b-instruct-ov<br>
LLM batch-size: 2<br>
ASR Model: distil-whisper/distil-small.en<br>
ASR load in 8bit: True<br>
ASR batch-size: 8<br>
Inference Device: GPU<br>

```
export LLM_MODEL=llmware/llama-3.2-3b-instruct-ov
export LLM_BATCH_SIZE=2
export ASR_MODEL=distil-whisper/distil-small.en
export ASR_LOAD_IN_8BIT=1
export ASR_BATCH_SIZE=8
export INF_DEVICE=GPU
./run-demo.sh sample.wav
```
