import onnxruntime
from torchvision import transforms
from PIL import Image as Image
import cv2
import numpy as np
def blur2clean():
    input_img = cv2.imread('model_deploy/onnx/face.png').astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1]) 
    input_img = np.expand_dims(input_img, 0)
    
    ort_session = onnxruntime.InferenceSession("model_deploy/onnx/srcnn_sim.onnx") 
    ort_inputs = {'input': input_img} 
    ort_output = ort_session.run(['output'], ort_inputs)[0] 
    
    ort_output = np.squeeze(ort_output, 0) 
    ort_output = np.clip(ort_output, 0, 255) 
    ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
    cv2.imwrite("model_deploy/onnx/face_ort.png", ort_output)
    print("success")
if __name__ == "__main__":
    blur2clean()