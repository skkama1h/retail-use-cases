import argparse
import time
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import OpenVINOSpeechToTextLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import docs_loader_utils as docs_loader
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ast

MAX_NEW_TOKENS = 5040

parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
parser.add_argument("--model_id", nargs="?", default="llmware/llama-3.2-3b-instruct-ov")
parser.add_argument("--asr_model_id", nargs="?", default="distil-whisper/distil-small.en")
parser.add_argument("--device", nargs="?", default="GPU")
parser.add_argument("--asr_batch_size", default=1, type=int)
parser.add_argument("--asr_load_in_8bit", default=False, action="store_true")
parser.add_argument("--k_means_enabled", default=False, action="store_true")
parser.add_argument("--llm_batch_size", default=2, type=int)
args = parser.parse_args()

print("LangChain OpenVINO Chapterization")
print("LLM model_id: ", args.model_id)
print("LLM batch_size: ", args.llm_batch_size)
print("ASR model_id: ", args.asr_model_id)
print("ASR batch_size: ", args.asr_batch_size)
print("ASR load_in_8bit: ", args.asr_load_in_8bit)
print("Inference device  : ", args.device)
print("K-Means enabled: ", args.k_means_enabled)
print("Audio file: ", args.audio_file)
#input("Press Enter to continue...")

print("Initializing....")
asr_loader = OpenVINOSpeechToTextLoader(args.audio_file, 
        args.asr_model_id, 
        device=args.device, 
        load_in_8bit=args.asr_load_in_8bit, 
        batch_size=args.asr_batch_size
)
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache-ov-models"}
ov_llm = HuggingFacePipeline.from_model_id(
    model_id=args.model_id,
    task="text-generation",
    backend="openvino",
    batch_size=args.llm_batch_size,
    model_kwargs={"device": args.device, "ov_config": ov_config},
    pipeline_kwargs={
        "max_new_tokens": MAX_NEW_TOKENS, 
        "do_sample": True, 
        "top_k": 10, 
        "temperature": 0.7, 
        "return_full_text": False, 
        "repetition_penalty": 1.0, 
        "encoder_repetition_penalty": 1.0, 
        "use_cache": True
    }
)
ov_llm.pipeline.tokenizer.pad_token_id = ov_llm.pipeline.tokenizer.eos_token_id
if args.k_means_enabled:
    from langchain_community.embeddings import OpenVINOEmbeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": args.device}
    encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}

    ov_embeddings = OpenVINOEmbeddings(
        model_name_or_path=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

print("Starting Auto-Chapter Creation")
start_time = time.time()
docs = asr_loader.load()
text = docs_loader.format_docs(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1000)

# transcript broken into overlapping chunks
docs = text_splitter.create_documents([text])

if args.k_means_enabled:
    # Convert to most relevant text in a single doc for single inference
    import numpy as np
    from sklearn.cluster import KMeans

    # 768 dimensional
    vectors = ov_embeddings.embed_documents([x.page_content for x in docs])
    num_clusters = 10 # TODO: adjust for smaller size text docs
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    docs = [docs[doc] for doc in selected_indices]

    # Note: this single batch method for large text is only supported 
    # through this method due to limited GPU/OpenCL memory allocation
    docs = [Document(page_content=docs_loader.format_docs(docs),
            metadata={
                "language": 'en',
                "summary" : '',
                "topic": '',
                "start": '',
                "end": ''
            })]

def get_summaries(transcript):
    for i in range(0, len(transcript)):
        doc_in = [transcript[i]]
        text = docs_loader.format_docs(doc_in)

        template = [{"role": "user", "content": f"Write a response that appropriately completes the request.\n\n### Instruction:\nYou are a helpful assistant. The provided text contains material from an educational lecture. Chapterize the text by providing the main topic sentences and short summaries of the material. The sections should be organized similar to the following example:\ntopic: \"Greetings and Class Introduction\"\nsummary: \"The teacher starts the session with greetings and pleasantries, creating a warm and positive atmosphere. By checking on students' well-being, the teacher fosters engagement and a sense of community, laying the groundwork for effective learning and connection.\"\n\n### Input:\n{text}\n\n### Response:\n"}]

        formatted_prompt = ov_llm.pipeline.tokenizer.apply_chat_template(template, tokenize=False)
        summary = ov_llm.invoke(formatted_prompt)
        summary = summary.replace('assistant\n\n', '')
        #print("Summaririze this: ", len(text), " ", text)
        #print("\n--Chapterization: ", summary)
        #print("-------------------------")
        #print('\n\n')
        transcript[i].metadata["summary"] = summary


print("-------Chapterization Results---------")
get_summaries(docs) #, chunk_size=1000) #61000) #5000)

for i in range(0, len(docs)):
    doc = docs[i]
    print(i+1, ") ", doc.metadata["summary"])
    print('\n')

end_time = time.time()
print("\n\n")
print("Total time taken for completion: ", end_time - start_time, " (seconds) / ", (end_time-start_time)/60, " (minutes)")
