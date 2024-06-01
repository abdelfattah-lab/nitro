import requests
import torch
from PIL import Image
from io import BytesIO
from optimum.intel.openvino import OVStableDiffusionImg2ImgPipeline, OVWeightQuantizationConfig
import os
model_id = "runwayml/stable-diffusion-v1-5"
device = "CPU"

quantization_config = OVWeightQuantizationConfig(bits=4)

if os.path.exists("ov_model_sd"):
    print("ov_model found.")
    pipeline = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        "ov_model_sd",
        device=device,
        quantization_config=quantization_config,
        compile=False
    )
else:
    print("ov_model not found.")
    pipeline = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        export=True,
        device=device,
        quantization_config=quantization_config,
        compile=False
    )
    pipeline.save_pretrained("ov_model_sd")

# Statically reshape the model for GPU
batch_size = 1
num_images_per_prompt = 1
height = 512
width = 768
pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
pipeline.compile()

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
prompt = "A fantasy landscape, trending on artstation"

image = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
image.save("fantasy_landscape-x.png")