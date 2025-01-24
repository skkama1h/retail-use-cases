import os
import torch
import argparse
from PIL import Image
from pydantic import Field
from typing import List, Optional
from decord import VideoReader, cpu
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModel

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

class MiniCPMV26Wrapper(LLM):
    model: object 
    tokenizer: object
    max_token_length: int = Field(default=128)

    @property
    def _llm_type(self) -> str:
        return "Custom OV MiniCPM-V-2_6"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:

        # Set decode params for video. TO DO. Make class attributes
        params={}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448

        # Format prompt for "model.chat()"
        video_fh, question = prompt.split(',')
        frames = encode_video(video_fh)
        msgs = [{'role': 'user', 'content': frames + [question]},]

        # Chat
        response = self.model.chat(
            image=None,
            msgs=msgs,
            max_length=self.max_token_length,
            tokenizer=self.tokenizer,
            **params
        )
        
        return response

if __name__ == '__main__':

    # Parse inputs
    parser_txt = "Generate video summarization using Langchanin, Openvino-genai, and MiniCPM-V-2_6."
    parser = argparse.ArgumentParser(parser_txt)
    parser.add_argument("-v", "--video_file", type=str,
                        help='Path to video you want to summarize.')
    parser.add_argument("-p", "--prompt", type=str,
                        help="Text prompt. By default set to: `Please summarize this video.`",
                        default="Please summarize this video.")
    args = parser.parse_args()
    if not os.path.exists(args.video_file):
        print(f"{args.video_file} does not exist.")
        exit()
        
    # Set globals model (To Do: make tokens actually used, make both flexible from cli)
    MAX_NEW_TOKENS = 5040 
    MAX_NUM_FRAMES = 64 # if OOM set a smaller number

    # Load Model (To Do: openvino-genai)
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6',
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6',
                                              trust_remote_code=True)

    # Wrap model in custom wrapper
    ovminicpm_wrapper = MiniCPMV26Wrapper(model=model,
                                          tokenizer=tokenizer,
                                          max_token_length=MAX_NEW_TOKENS)

    # Create template for inputs
    prompt = PromptTemplate(
        input_variables=["video", "question"],
        template="{video},{question}"
    )
    
    # Create pipeline and invoke
    chain =  prompt | ovminicpm_wrapper
    inputs = {"video": args.video_file, "question": args.prompt}    
    output = chain.invoke(inputs)
    print(output)
