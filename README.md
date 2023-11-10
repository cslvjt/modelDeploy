# modelDeploy
描述：将SRCNN打包为ncnn可调用格式
# 环境介绍
+ 操作系统：Ubuntu 18.04
+ python 3.6
# 所需库
see requirements.txt
```
pip install -r requirements.txt
```
# 模型部署流程
## model+pth ---> onnx 
首先将模型转化为onnx

```python model2onnx.py```

接着应用onnxsim将模型简化，以srcnn.onnx为例

```python3 -m onnxsim srcnn.onnx srcnn_sim.onnx```
## onnx ---> ncnn


### 使用ncnn框架部署
+ 建议按照该文章安装ncnn框架 https://zhuanlan.zhihu.com/p/506889381
+ 使用该网站进行线上转换 https://convertmodel.com/

# 参考
+ SRCNN代码和权重：https://zhuanlan.zhihu.com/p/477743341
+ gradio: https://www.gradio.app/guides/quickstart