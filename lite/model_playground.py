
from rewritten_models import *
import torch
from config import args, input_shapes, inputs # PARAMS DEFINED HERE
import time
from nncf import compress_weights, CompressWeightsMode
import psutil
import threading, multiprocessing

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024 * 1024)  # Convert bytes to MB

def monitor_memory_usage(interval, stop_event, max_memory_usage):
    current_max = 0
    while not stop_event.is_set():
        memory_usage = get_memory_usage()
        if memory_usage > current_max:
            current_max = memory_usage
        time.sleep(interval)
    max_memory_usage.value = current_max

if __name__ == "__main__":
    model = Transformer(args)

    # Loading stuff
    checkpoint = "/home/abdelfattah/llama3/Meta-Llama-3-8B/consolidated.00.pth"
    state = torch.load(checkpoint)
    model.load_state_dict(state, strict=False)

    a, b, c = model(**inputs)

    # import openvino as ov
    # import openvino.runtime.passes as passes

    # core = ov.Core()
    # ov_model = ov.convert_model(model, example_input=inputs, input=input_shapes)
    # ov.save_model(ov_model, "llama-lite.xml")
    # del model

    # # weight compression
    # # ov_model = compress_weights(ov_model, mode=CompressWeightsMode.INT8_ASYM)

    # print("Starting compilation benchmarks")

    # for device in ["CPU", "GPU", "NPU"]:
    #     mem_before_compilation = get_memory_usage()

    #     print(f"\n ---- START MEMORY: {mem_before_compilation} ---- \n")

    #     # COMPILATION ####
    #     max_memory_usage = multiprocessing.Value('d', 0.0)
    #     stop_event = threading.Event()
    #     monitor_thread = threading.Thread(target=monitor_memory_usage, args=(0.1, stop_event, max_memory_usage))

    #     start = time.time()
    #     monitor_thread.start()

    #     model = ov.compile_model(ov_model, device_name=device)
        
    #     compile_time = time.time() - start
    #     stop_event.set()
    #     monitor_thread.join()
    #     peak_memory = max_memory_usage.value

    #     print(f"Compiled time on {device} > {compile_time} seconds")
    #     print(f"Peak memory usage on {device} > {peak_memory - mem_before_compilation} GB\n")

    #     # INFERENCE ####
    #     start = time.time()
    #     for i in range(20):
    #         output = model(inputs)
    #     print(f"Inference time on {device} > {time.time() - start} seconds\n")

    #     del model