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
    max_length: int = Field(default=128)

    @property
    def _llm_type(self) -> str:
        return "Custom MiniCPM-V-2_6"

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
        prompt = prompt.split(',')
        frames = encode_video(prompt[0])
        question = prompt[-1]        
        msgs = [{'role': 'user', 'content': frames + [question]},]

        # Chat
        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        
        return str(response)

if __name__ == '__main__':

    # Parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    parser.add_argument("prompt")
    args = parser.parse_args()
    
    # Set globals model (To Do: make flexible from cli)
    MAX_NEW_TOKENS = 5040
    MAX_NUM_FRAMES = 64 # if OOM set a smaller number

    # Load Model
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6',
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6',
                                              trust_remote_code=True)

    # Wrap model in custom wrapper
    minicpm_llm = MiniCPMV26Wrapper(model=model,
                                    tokenizer=tokenizer,
                                    max_length=MAX_NEW_TOKENS)

    # Create template for inputs
    prompt = PromptTemplate(
        input_variables=["video", "question"],
        template="{video},{question}"
    )
    
    # Create pipeline and invoke
    chain =  prompt | minicpm_llm
    inputs = {"video": args.video_file, "question": args.prompt}    
    output = chain.invoke(inputs)
    print(output)
