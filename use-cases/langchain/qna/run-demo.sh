#!/bin/bash

source langchain_audio_qna_env/bin/activate
export TOKENIZERS_PARALLELISM=true

INPUT_FILE=$1
ASR_8BIT_ENABLE_FLAG="--asr_load_in_8bit"
ENABLE_FLAGS=""

if [ "$ASR_LOAD_IN_8BIT" == "1" ]
then
	ENABLE_FLAGS=$ASR_8BIT_ENABLE_FLAG
fi

#python3 test-tts.py
#exit 0
echo "Run audio qna"
python3 audio_qna.py $INPUT_FILE --model_id $LLM_MODEL --device $INF_DEVICE --asr_batch_size $ASR_BATCH_SIZE --llm_batch_size $LLM_BATCH_SIZE --asr_model_id $ASR_MODEL $ENABLE_FLAGS
