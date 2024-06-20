from config import *
import subprocess
from optimum.intel import OVModelForCausalLM

ov_model = OVModelForCausalLM.from_pretrained(HF_MODEL_NAME, export=True, compile=False)

ov_model.save_pretrained("models/cli-model")