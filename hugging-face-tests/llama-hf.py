from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from optimum.intel.openvino import OVModelForCausalLM
import os
from collections import defaultdict
import openvino as ov
import openvino.runtime.passes as passes

def visualize_example(m:ov.Model, file:str='image'):
    """
    Produces a visualization in .dot format. Viewable from a dot loader, e.g.
    https://dreampuf.github.io/GraphvizOnline/
    """
    pass_manager = passes.Manager()
    pass_manager.register_pass(passes.VisualizeTree(file_name=file))
    pass_manager.run_passes(m)

model_id = "meta-llama/Meta-Llama-3-8B"
device = "HETERO:GPU,CPU"
model_ir = "ov_model"

# check if the folder exists
if not os.path.exists("models"):
    os.mkdir("models")

model_internal = os.path.join("models", model_ir)
# Load model if it exists, or generate model and save it

core = ov.Core()
if os.path.exists(model_internal):
    print("ov_model found.")
    model = core.read_model("./models/ov_model/openvino_model.xml")
else:
    print("ov_model not found.")
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, device=device,)
    model.save_pretrained(model_internal)

tokenizer=AutoTokenizer.from_pretrained(model_id)

# counter = defaultdict(int)
# for node in model.get_ops():
#     counter[node.get_type_name()] += 1

# for k in counter:
#     print(f"{k}: {counter[k]}")
visualize_example(model, "a.svg")

# Generate a pipeline
# pipe = pipeline(
#     task="text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

# k = pipe(
#     "How is your day today?",
#     max_new_tokens = 128,
#     eos_token_id=tokenizer.eos_token_id
#   )
# print(k)
