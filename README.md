# MTCNN face detection using tensorflow end2end

### Introduction
This is a demo for MTCNN using TensorFlow api end2end.For more details, please refer to the paper [arXiv paper](https://arxiv.org/pdf/1604.02878.pdf).

### Dependencies
* Tensorflow 1.4.1
* TF-Slim
* Python 3.6
* Ubuntu 16.04
* Cuda 8.0

### Usage
First you should run 'python npy2ckpt.py' to convert the three npy file for pnet/rnet/onet to one ckpt if you do not have the ckpt file(Note:the three npy files and converted ckpt file already in mtcnn_model of this repository).

Then Replace your picture in 'examples' and run 'python demo.py'.

the result is  'examples/*_result.jpg'.

### Result

demo_result:

<img align="right" src="https://github.com/wanjinchang/MTCNN_USE_TF_E2E/blob/master/examples/1_result.jpg">

<img align="right" src="https://github.com/wanjinchang/MTCNN_USE_TF_E2E/blob/master/examples/2_result.jpg">

<img align="right" src="https://github.com/wanjinchang/MTCNN_USE_TF_E2E/blob/master/examples/img_140_result.jpg">

<img align="right" src="https://github.com/wanjinchang/MTCNN_USE_TF_E2E/blob/master/examples/img_414_result.jpg">

### References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter(https://arxiv.org/pdf/1604.02878.pdf).
2. [facenet](https://github.com/davidsandberg/facenet)



