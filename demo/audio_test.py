from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor,Qwen2AudioProcessor
import torch
cuda0 = torch.device('cuda:0')

processor = Qwen2AudioProcessor.from_pretrained("/disk0/aitogether/ai-english/yi_model/model/qwen/qwen2-audio-7b-instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("/disk0/aitogether/ai-english/yi_model/model/qwen/qwen2-audio-7b-instruct", torch_dtype = "auto",device_map="auto").eval()

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}, 
    {"role": "user", "content": [
        {"type": "audio", "audio": "/disk0/aitogether/xishaojian/audio_lab/audio_resources/PPT+Teacher+Student.MP3"},
        {"type":"text","text":"里面有多个说话者，请提取每个说话者对应音频里面的内容"}
    ]},
]
text = processor.apply_chat_template(conversation,chat_template = processor.default_chat_template, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                if "audio_url" in ele:
                    audios.append(
                        librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)[0]
                    )
                elif "audio" in ele:
                    audios.append(
                        librosa.load(ele['audio'], sr=processor.feature_extractor.sampling_rate)[0]
                    )
print(text)
inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
print(inputs)
inputs["input_ids"] = inputs.input_ids.to("cuda")
for k,v in inputs.items():
        file_name = "./tensordata_1/"+k+".pt"
        torch.save(v,file_name)
generate_ids = model.generate(**inputs, max_length=4096)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]
print(generate_ids)
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)
