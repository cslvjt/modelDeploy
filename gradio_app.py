import gradio as gr
import cv2
def superResolution(img_path,factor):
    img = cv2.imread(img_path)
    w = int(img.shape[0] * factor)
    h = int(img.shape[1] * factor)
    dim = (w,h)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype("float32")
    # output = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    output = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
    return output

def deBlur(img_path):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype("float32")
    output = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return output
    

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown(" image restoration")
        with gr.Tab("superResolution"):
            sr_input = gr.Image(type="filepath", label="Input")
            sr_factor = gr.Number(value=2,label="Upscaling factor (up to 4)")
            sr_output = gr.Image(type="numpy", label="Output")
            sr_button = gr.Button("superResolution")
        with gr.Tab("image deblur"):
            blur_input = gr.Image(type="filepath", label="Input")
            blur_output = gr.Image(type="numpy", label="Input")
            blur_button = gr.Button("deblur")

        sr_button.click(
            fn=superResolution,
            inputs=[sr_input,sr_factor],
            outputs=[sr_output]
        )
        
        blur_button.click(
            fn=deBlur,
            inputs=[blur_input],
            outputs=[blur_output]
        )
    demo.launch()