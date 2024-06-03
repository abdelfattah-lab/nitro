from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import requests, PIL, io, torch
# Get a picture of a cat from the web:
img = PIL.Image.open(io.BytesIO(requests.get("https://placekitten.com/200/300").content))

# Torchvision model and input data preparation from https://pytorch.org/vision/stable/models.html
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)
# PyTorch model inference and post-processing
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}% (with PyTorch)")

# OpenVINO model preparation and inference with the same post-processing
import openvino as ov
converted_model = ov.convert_model(model, example_input=batch)
compiled_model = ov.compile_model(converted_model, device_name="NPU")
runtime_info = compiled_model.get_runtime_model()
prediction = torch.tensor(compiled_model(batch)[0]).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}% (with OpenVINO)")