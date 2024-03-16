import torch
from model.srcnn_arch import load_net
from model.LWDNet_arch import load_LWDNet
import argparse

def model2onnx(args):
    # TODO: 如何自定义适应输入分辨率
    inputTensor = torch.randn(1, 3, args.h, args.w)

    if args.model_name == "SRCNN":
        net = load_net(args.weight_path)
    elif args.model_name == "LWDNet":
        net = load_LWDNet(args.weight_path)
    else:
        raise ValueError(f"{args.model_name} not find")
    with torch.no_grad():
        torch.onnx.export(
            net,
            inputTensor,
            args.onnx_path,
            opset_version=11, 
            input_names=['input'], 
            output_names=['output']
        )
    print(args.onnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=256)
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    print(args)
    model2onnx(args)