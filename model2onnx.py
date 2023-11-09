import torch
from SRCNN.srcnn_arch import load_net

def model2onnx(onnx_path):
    # TODO: 如何自定义适应输入分辨率
    inputTensor = torch.randn(1, 3, 256, 256) 
    net = load_net()
    with torch.no_grad():
        torch.onnx.export(
            net,
            inputTensor,
            onnx_path,
            opset_version=11, 
            input_names=['input'], 
            output_names=['output']
        )

if __name__ == "__main__":
    model2onnx("SRCNN/srcnn.onnx")