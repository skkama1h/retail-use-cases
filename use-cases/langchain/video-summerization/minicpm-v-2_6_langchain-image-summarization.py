import re
import cv2
import torch
import base64
import argparse
from PIL import Image
from io import BytesIO
from pydantic import Field
from typing import List, Optional
from langchain.llms.base import LLM
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModel

def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def decode_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

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

        # Format prompt for model.chat format. TO DO: CLEAN UP!! Currently ignoring system prompt too.
        prompt = re.split(':|\n', prompt)
        image = decode_image(prompt[-3])
        question = prompt[-1]
        msgs = [{'role': 'user', 'content': [image, question]}]

        # Invoke model 
        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        return str(response)

if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file")
    parser.add_argument("prompt")
    args = parser.parse_args()
    
    # Load model   
    MAX_NEW_TOKENS = 5040
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6',
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6',
                                              trust_remote_code=True)
    minicpm_llm = MiniCPMV26Wrapper(model=model,
                                    tokenizer=tokenizer,
                                    max_length=MAX_NEW_TOKENS)

    # ChatPromptTemplate setup
    prompt = ChatPromptTemplate.from_messages([
        ('system', "You are a helpful assistant capable of answering questions about images."),
        ('user', "Here is an image and a question:\nImage: {image}\nQuestion: {question}")
    ])

    # Create the pipeline
    image = Image.open(args.image_file)

    # Prepare inputs
    base64_image = encode_image(image)
    inputs = {"image": base64_image, "question": args.prompt}

    # Create and invoke
    chain =  prompt | minicpm_llm
    output = chain.invoke(inputs)
    print(output)


    ##### Run model directly
    # image = Image.open('my_image.png').convert('RGB')
    # question = 'What is in the image?'
    # msgs = [{'role': 'user', 'content': [image, question]}]

    # res = model.chat(
    #     image=None,
    #     msgs=msgs,
    #     tokenizer=tokenizer
    # )
    # print(res)
