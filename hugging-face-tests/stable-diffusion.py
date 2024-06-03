import requests
import torch
from PIL import Image
from io import BytesIO
from optimum.intel import OVStableDiffusionImg2ImgPipeline
import os

model_id = "runwayml/stable-diffusion-v1-5"
DEVICE = "GPU"
PATH = "models/ov_model_sd"

if os.path.exists(PATH):
    print("ov_model found.")
    pipeline = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        model_id  = PATH,
        device    = DEVICE,
    )
else:
    print("ov_model not found.")
    pipeline = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        model_id  = model_id,
        device    = DEVICE,
        export    = True,
    )
    pipeline.save_pretrained(PATH)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
prompt = "A fantasy landscape, trending on artstation"

image = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
image.save("fantasy_landscape-x.png")