import os
import pathlib

# Parameters
TOKEN_SEQUENCE_LENGTH = 32
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL = "llama3_onnx"

# Which model in the models/folder you would like to use
REPO_PATH = pathlib.Path(os.path.abspath(__file__)).parent
ROOT_DIRECTORY = f"{REPO_PATH}/models/"
MODEL_DIRECTORY = f"{ROOT_DIRECTORY}/{MODEL}"
TOKENIZER_DIR = MODEL_DIRECTORY