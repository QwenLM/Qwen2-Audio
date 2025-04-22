from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import librosa
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
import torch
from io import BytesIO

app = FastAPI()

# 初始化模型和处理器
processor = Qwen2AudioProcessor.from_pretrained("/disk0/aitogether/ai-english/yi_model/model/qwen/qwen2-audio-7b-instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("/disk0/aitogether/ai-english/yi_model/model/qwen/qwen2-audio-7b-instruct", torch_dtype="auto", device_map="auto").eval()

class Conversations(BaseModel):
    conver: list
    

def analyse_audio(conversation) -> str:

    text = processor.apply_chat_template(conversation, chat_template=processor.default_chat_template, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(ele['audio'], sr=processor.feature_extractor.sampling_rate)[0]
                    )
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs["input_ids"] = inputs.input_ids.to("cuda")
    generate_ids = model.generate(**inputs, max_length=4096)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

@app.post("/analyse")
async def analyse(input_data:Conversations):
    print(input_data)
    res = analyse_audio(input_data.conver)
    print(type(res))
    return {'response':res}
    # response = analyse_audio(query=query, audio=audio)
    # return {"response": response}

