import os
import time
import torch
import argparse
import numpy as np
import openvino_genai
from PIL import Image
from pydantic import Field
from openvino import Tensor
from typing import List, Optional
from decord import VideoReader, cpu
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModel
from VideoChunkLoader import VideoChunkLoader

def encode_video(video_path, max_num_frames=64, resolution=[]):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    
    if len(resolution) != 0:
        vr = VideoReader(video_path, width=resolution[0],
                         height=resolution[1], ctx=cpu(0))
    else:
        vr = VideoReader(video_path, ctx=cpu(0))
        
    # frame_idx = [i for i in range(0, len(vr), sample_fps)]
    frame_idx = [i for i in range(0, len(vr), int(len(vr)/max_num_frames))]    
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Tensor(v.astype('uint8')) for v in frames]
    print('Num frames sampled:', len(frames))
    return frames

def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return False".

class OVMiniCPMV26Wrapper(LLM):
    ovpipe: object 
    generation_config: object
    max_num_frames: int
    resolution: list[int]
    
    @property
    def _llm_type(self) -> str:
        return "Custom OV MiniCPM-V-2_6"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:

        video_fh, question = prompt.split(',', 1)
        frames = encode_video(video_fh, self.max_num_frames,
                              resolution=self.resolution)        
        self.ovpipe.start_chat()
        self.ovpipe.generate(question,
                             images=frames,
                             generation_config=self.generation_config,
                             streamer=streamer)
        self.ovpipe.finish_chat()
        
        return ''

if __name__ == '__main__':

    # Parse inputs
    parser_txt = "Generate video summarization using Langchanin, Openvino-genai, and MiniCPM-V-2_6."
    parser = argparse.ArgumentParser(parser_txt)
    parser.add_argument("video_file", type=str,
                        help='Path to video you want to summarize.')
    parser.add_argument("model_dir", type=str,
                        help="Path to openvino-genai optimized model")    
    parser.add_argument("-p", "--prompt", type=str,
                        help="Text prompt. By default set to: `Please summarize this video.`",
                        default="Please summarize this video.")
    parser.add_argument("-d", "--device", type=str,
                        help="Target device for running ov MiniCPM-v-2_6",
                        default="CPU")    
    parser.add_argument("-t", "--max_new_tokens", type=int,
                        help="Maximum number of tokens to be generated.",
                        default=5040)
    parser.add_argument("-f", "--max_num_frames", type=int,
                        help="Maximum number of frames to be sampled per chunk for inference. Set to a smaller number if OOM.",
                        default=64)
    parser.add_argument("-c", "--chunk_duration", type=int,
                        help="Maximum length in seconds for each chunk of video.",
                        default=90)
    parser.add_argument("-o", "--chunk_overlap", type=int,
                        help="Overlap in seconds beteen chunks of input video.",
                        default=2)
    parser.add_argument("-r", "--resolution", type=int, nargs=2,
                        help="Desired spatial resolution of input video if different than original. Width x Height")

    tot_st_time = time.time()
    args = parser.parse_args()
    if not os.path.exists(args.video_file):
        print(f"{args.video_file} does not exist.")
        exit()

    # Load model
    enable_compile_cache = dict()
    if "GPU" == args.device:
        # Cache compiled models on disk for GPU to save time on the
        # next run. It's not beneficial for CPU.
        enable_compile_cache["CACHE_DIR"] = "vlm_cache"
    pipe = openvino_genai.VLMPipeline(args.model_dir, args.device, **enable_compile_cache)

    # Set variables for inference 
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    
    # Wrap model in custom wrapper
    resolution = [] if not args.resolution else args.resolution
    ovminicpm_wrapper = OVMiniCPMV26Wrapper(ovpipe=pipe,
                                            generation_config=config,
                                            max_num_frames=args.max_num_frames,
                                            resolution=resolution)

    # Create template for inputs
    prompt = PromptTemplate(
        input_variables=["video", "question"],
        template="{video},{question}"
    )
    
    # Create pipeline and invoke
    chain =  prompt | ovminicpm_wrapper

    # Load video create docs
    loader = VideoChunkLoader(
        video_path=args.video_file,
        chunking_mechanism="sliding_window",
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap
        # specific_intervals=[
        #     {"start": 5, "duration": 10},
        #     {"start": 20, "duration": 8},
        # ],
    )
    
    for doc in loader.lazy_load():
        # print(f"Chunk Metadata: {doc.metadata}")
        print(f"Chunk Content: {doc.page_content}")
        
        # Loop through docs
        chunk_st_time = time.time()
        inputs = {"video": doc.metadata['chunk_path'], "question": args.prompt}        
        output = chain.invoke(inputs)
        print("\nChunk Inference time: {} sec\n".format(time.time() - chunk_st_time))
    print("\nTotal Inference time: {} sec\n".format(time.time() - tot_st_time))

    
