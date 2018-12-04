# MTCNN face detection & alignment all in TensorFlow


### Introduction
This is a demo for MTCNN implementation all in TensorFlow api to take advantage of GPU computing resource.For more details of MTCNN, please refer to the paper [arXiv paper](https://arxiv.org/pdf/1604.02878.pdf).

### Dependencies
* TensorFlow 1.4.1
* TF-Slim
* Python 3.6
* Ubuntu 16.04
* Cuda 8.0

### Usage
First you should run 'python npy2ckpt.py' to convert the three npy files(get from [facenet](https://github.com/davidsandberg/facenet)) for pnet/rnet/onet to one checkpoint if you do not have the checkpoint file(Note:the three npy files and converted checkpoint file already in mtcnn_model of this repository).

Then replace your pictures in 'examples' and run 'python demo.py'.

### Result

demo_result:

<div align=center><img src="https://github.com/wanjinchang/MTCNN_USE_TF_E2E/blob/master/examples/1_result.jpg"/></div>

<div align=center><img src="https://github.com/wanjinchang/MTCNN_USE_TF_E2E/blob/master/examples/img_140_result.jpg"/></div>

<div align=center><img src="https://github.com/wanjinchang/MTCNN_USE_TF_E2E/blob/master/examples/img_414_result.jpg"/></div>

&nbsp;
&nbsp;

### References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter(https://arxiv.org/pdf/1604.02878.pdf).
2. [facenet](https://github.com/davidsandberg/facenet)



