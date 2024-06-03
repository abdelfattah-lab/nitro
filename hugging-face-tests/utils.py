import os
from optimum.intel.openvino.modeling import OVBaseModel

def print_bold(text):
    # ANSI escape code for bold
    print(f"\033[1m{text}\033[0m")

def load_model(model_id:str=None,
               ir_path:str=None,
               model:OVBaseModel=None,
               device:str="CPU",
               *args):
    """
    Creates a model using *args.
    """
    
    if os.path.exists(ir_path):
        print("OV Model at path found.")
        pipeline = model.from_pretrained(
            model_id  = ir_path,
            device    = device,
            compile   = False
        )
    else:
        print("OV Model at path found.")
        pipeline = model.from_pretrained(
            model_id  = model_id,
            device    = device,
            export    = True,
            compile   = False
        )
        pipeline.save_pretrained(ir_path)