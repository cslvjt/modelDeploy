import gradio as gr
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import utils
import numpy as np

class SuperResolutionNet(nn.Module): 
    def __init__(self, upscale_factor=2): 
        super().__init__() 
        self.upscale_factor = upscale_factor 
        self.img_upsampler = nn.Upsample( 
            scale_factor=self.upscale_factor, 
            mode='bicubic', 
            align_corners=False) 
 
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4) 
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0) 
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2) 
 
        self.relu = nn.ReLU() 
 
    def forward(self, x, upscale_factor=2): 
        x = F.interpolate(x,scale_factor=upscale_factor)
        # x = self.img_upsampler(x)
        
        out = self.relu(self.conv1(x)) 
        out = self.relu(self.conv2(out)) 
        out = self.conv3(out) 
        return out 
    
def load_net(weight_path= "SRCNN/srcnn.pth"):
    net = SuperResolutionNet()
    state_dict = torch.load(weight_path)['state_dict']
    # Adapt the checkpoint 
    for old_key in list(state_dict.keys()): 
        new_key = '.'.join(old_key.split('.')[1:]) 
        state_dict[new_key] = state_dict.pop(old_key) 
    net.load_state_dict(state_dict)
    net.eval()
    return net

def inference(img_path,upscale_factor):
    upscale_factor = int(upscale_factor)
    net = load_net()
    img = utils.read_img(str(img_path))
    out = net(img.unsqueeze(0))
    out = out[0].detach().numpy()
    out = rearrange(out,"c h w -> h w c")
    out = (out * 255.0).round().astype(np.uint8)
    return out


def to_black(img):
    output = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return output

if __name__ == "__main__":
    interface = gr.Interface(inference,
                            inputs=[gr.Image(type="filepath", label="Input"),
                            gr.Number(label="Upscaling factor (up to 4)")], # 上传一张图像，gradio会将其转化为numpy array格式 shape[h,w,c]
                            outputs=[gr.Image(type="numpy", label="Output")])
    interface.launch()