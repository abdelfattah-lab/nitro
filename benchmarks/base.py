class Benchmark:
    def __init__(self, model, name):
        self.model = model
        self.name = name
    
    def compile(self):
        pass

    def infer(self, input_shape):
        pass

    def __call__(self,
                 input_shape,
                 devices=["CPU", "GPU", "NPU"]):
        self.compile()
        self.infer(input_shape)