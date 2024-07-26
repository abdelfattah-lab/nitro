import openvino as ov
from typing import Optional

class Converter():
    """
    Class to convert provided PyTorch models into OpenVINO IR formats.
    """

    def __init__(model:str):
        pass

    def convert(self,
                dest:Optional[str] = None) -> ov.Model:
        pass

    def _stateful_model(self, suffix_match:str="_out") -> ov.Model:
        """
        Converts the models into stateful models. Stateful models are created by
        matching parameter names [x] with result names [x{suffix_match}]. The
        shapes must be the same.
        """
        pass

    def _generate_example_inputs_for_conversion():
        pass

    def _rename():
        pass