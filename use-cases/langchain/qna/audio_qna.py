import argparse
import time
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import OpenVINOSpeechToTextLoader
from langchain_community.tools import OpenVINOText2SpeechTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import docs_loader_utils as docs_loader
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ast

MAX_NEW_TOKENS = 512

parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
parser.add_argument("--model_id", nargs="?", default="llmware/llama-3.2-3b-instruct-ov")
parser.add_argument("--asr_model_id", nargs="?", default="distil-whisper/distil-small.en")
parser.add_argument("--tts_model_id", nargs="?", default="OuteAI/OuteTTS-0.1-350M")
parser.add_argument("--device", nargs="?", default="GPU")
parser.add_argument("--asr_batch_size", default=1, type=int)
parser.add_argument("--asr_load_in_8bit", default=False, action="store_true")
parser.add_argument("--llm_batch_size", default=1, type=int)
args = parser.parse_args()

print("LangChain OpenVINO Audio QnA")
print("LLM model_id: ", args.model_id)
print("LLM batch_size: ", args.llm_batch_size)
print("ASR model_id: ", args.asr_model_id)
print("ASR batch_size: ", args.asr_batch_size)
print("ASR load_in_8bit: ", args.asr_load_in_8bit)
print("TTS model_id: ", args.tts_model_id)
print("Inference device  : ", args.device)
print("Audio file: ", args.audio_file)
#input("Press Enter to continue...")

asr_loader = OpenVINOSpeechToTextLoader(args.audio_file, 
        args.asr_model_id, 
        device=args.device, 
        load_in_8bit=args.asr_load_in_8bit, 
        batch_size=args.asr_batch_size
)
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache-ov-model"}

ov_llm = HuggingFacePipeline.from_model_id(
    model_id=args.model_id,
    task="text-generation",
    backend="openvino",
    batch_size=args.llm_batch_size,
    model_kwargs={"device": args.device, "ov_config": ov_config},
    pipeline_kwargs={"max_new_tokens": MAX_NEW_TOKENS, "do_sample": True, "top_k": 10, "temperature": 0.7, "return_full_text": False, "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0, "use_cache": True},
)
ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id

tts = OpenVINOText2SpeechTool(model_id="OuteAI/OuteTTS-0.1-350M",
        device="CPU", 
        load_in_8bit=True)

start_time = time.time()
docs = asr_loader.load()
text = docs_loader.format_docs(docs)

template = f"""Answer the user's question conscisely.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {text}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

print("Question asked: ", text)
text_to_speak = ov_llm.invoke(template)

print("TTS to play answer: ", text_to_speak)

## tts
speech = tts.run(text_to_speak)
print("Playing speech file: ", speech)
tts.play(speech)
end_time = time.time()
print("\n\n")
print("Total time taken for completion: ", end_time - start_time, " (seconds) / ", (end_time-start_time)/60, " (minutes)")
