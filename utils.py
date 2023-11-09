import torchvision.transforms.functional as tf
import torch
import cv2




def read_img(img_path):
    """
    return:
        c h w
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype("float32")
    img = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
    return img

def tensor2img(tensor,name):
    """
    tensor:
        c h w
    """
    image = tf.to_pil_image(tensor)
    image.save(name)