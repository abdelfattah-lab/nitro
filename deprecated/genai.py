import openvino_genai as ov_genai

model_path = "models/llama3_optimum/"

pipe = ov_genai.LLMPipeline(model_path, "CPU")
print(pipe.generate("The Sun is yellow because"))