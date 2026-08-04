import torch

class OnnxExporter:
    def __init__(self, filename, classifier):
        self.filename = filename
        self.classifier = classifier

    def export(self, input):
        path = "onnx_models/" + self.filename
        torch.onnx.export(
            self.classifier.model,
            input,
            path,
            external_data=False
        )

        print(f"Saved to {path}")