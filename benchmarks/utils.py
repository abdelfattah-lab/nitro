# Random helper functions
import openvino as ov
import openvino.runtime.passes as passes

def probe(model:ov.Model, probe_enabled:bool=True, *args):
    """
    Probe the attributes of an OV model, such as inputs and outputs.
    *args are the list of attributes you want to view.
    """
    if not probe_enabled: return
    for a in args:
        print(f"{a}:", model.__getattribute__(a))

def visualize_example(m:ov.Model, file:str='image'):
    """
    Produces a visualization in .dot format. Viewable from a dot loader, e.g.
    https://dreampuf.github.io/GraphvizOnline/
    """
    pass_manager = passes.Manager()
    pass_manager.register_pass(passes.VisualizeTree(file_name=file))
    pass_manager.run_passes(m)