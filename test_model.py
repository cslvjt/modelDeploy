from model.srcnn_arch import load_net
import utils
import numpy as np
import onnxruntime
import cv2
import argparse

def load_model(model_name, weight_path):
    if model_name == "SRCNN":
        net = load_net(weight_path)
    else:
        raise ValueError(f"{model_name} not find")
    return net
    
def test_model(model, img_path, output_path):
    img = utils.read_img(img_path)
    out = model(img.unsqueeze(0))
    utils.tensor2img(out[0], output_path)

def test_onnx(onnx_model, img_path, output_path):
    input_img = cv2.imread(img_path).astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1]) 
    input_img = np.expand_dims(input_img, 0)
    
    ort_session = onnxruntime.InferenceSession(onnx_model,providers=["CPUExecutionProvider"]) 
    ort_inputs = {'input': input_img} 
    ort_output = ort_session.run(['output'], ort_inputs)[0] 
    
    ort_output = np.squeeze(ort_output, 0) 
    ort_output = np.clip(ort_output, 0, 255) 
    ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
    cv2.imwrite(output_path, ort_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="weights/srcnn.pth")
    parser.add_argument("--onnx_path", type=str, default="onnx_weight/srcnn_256X256.onnx")
    parser.add_argument("--model_name", type=str, default="SRCNN")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    print(args)
    model = load_model(args.model_name, args.weight_path)
    test_model(model, args.img_path, args.output_path)
    test_onnx(args.onnx_path, args.img_path, args.output_path)