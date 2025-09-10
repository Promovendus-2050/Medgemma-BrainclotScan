# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
#https://radiopaedia.org/cases/intracerebral-haemorrhage-warfarinised?lang=us
# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
# image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
# image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
image0 = Image.open("case1/0.png")
image1 = Image.open("case1/1.png")
image2 = Image.open("case1/2.png")
image3 = Image.open("case1/3.png")
image4 = Image.open("case1/4.png")
image5 = Image.open("case1/5.png")
image6 = Image.open("case1/6.png")

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Elderly male patient. Can you analyze these 7 CT scans?"},
            {"type": "image", "image": image0},
            {"type": "image", "image": image1},
            {"type": "image", "image": image2},
            {"type": "image", "image": image3},
            {"type": "image", "image": image4},
            {"type": "image", "image": image5},
            {"type": "image", "image": image6},
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
