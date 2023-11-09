from SRCNN.srcnn_arch import load_net
import utils
import numpy as np
import onnxruntime
import cv2

def test_model(img_path,weight_path = "SRCNN/srcnn.pth"):
    net = load_net(weight_path)
    img = utils.read_img(img_path)
    out = net(img.unsqueeze(0))
    utils.tensor2img(out[0],"sr_result.png")

def test_onnx(img_path,onnx_path):
    input_img = cv2.imread(img_path).astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1]) 
    input_img = np.expand_dims(input_img, 0)
    
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=["CPUExecutionProvider"]) 
    ort_inputs = {'input': input_img} 
    ort_output = ort_session.run(['output'], ort_inputs)[0] 
    
    ort_output = np.squeeze(ort_output, 0) 
    ort_output = np.clip(ort_output, 0, 255) 
    ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
    cv2.imwrite("face_ort.png", ort_output)

if __name__ == "__main__":
    img_path = "image/face.png"
    onnx_path = "SRCNN/srcnn_sim.onnx"
    # test_model(img_path)
    test_onnx(img_path,onnx_path)