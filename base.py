from pathlib import Path

class Args:
    def __init__(self, args):
        for key, value in args.items():
            setattr(self, key, value)

class OVChunk:
    """
    Small wrapper for the OpenVINO model. Abstracts and simplifies the chunks to
    the following:

    - inputs (input + caches)
    - output (to be fed into the next chunk)
    - output caches
    """

    def __init__(self,
                 model_dir: Path | str):
        # TODO
        pass

class LLMBase:
    """
    Base model for LLM processing.
    """

    @classmethod
    def from_pretrained(cls,
                        max_batch_size:int,
                        n_sub_layers:int,
                        ) -> "LLMBase":
        """
        Generates the OpenVINO models, and initializes the script.
        """
        raise NotImplementedError("Not implemented - must be specified in sub-classes.")