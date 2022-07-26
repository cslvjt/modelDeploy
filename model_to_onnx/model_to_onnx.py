import torch
from SRCNN import init_torch_model
x = torch.randn(1, 3, 256, 256) 

model=init_torch_model()
with torch.no_grad(): 
    torch.onnx.export( 
        model, 
        x, 
        "srcnn.onnx", 
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'])