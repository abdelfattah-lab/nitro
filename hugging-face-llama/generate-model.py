from config import *
import subprocess

command = [
    "optimum-cli", 
    "export", 
    "openvino", 
    "--model", 
    HF_MODEL_NAME, 
    MODEL_DIR
]

result = subprocess.run(command, check=True, text=True, capture_output=True)