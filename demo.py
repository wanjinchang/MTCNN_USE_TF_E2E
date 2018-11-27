#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: demo.py
@time: 18-10-9 下午2:53
@desc:
'''
import tensorflow as tf
from detect_face_tf import Network
import cv2
import numpy as np
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
f_path = os.path.dirname(os.path.realpath(__file__))

tfmodel = './mtcnn_model/mtcnn_ckpt/mtcnn.ckpt'
# set config
tfconfig = tf.ConfigProto(allow_soft_placement=True)
# tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.07
tfconfig.gpu_options.allow_growth = True

# init session
sess = tf.Session(config=tfconfig)
net = Network(minsize=50, threshold=[0.6,0.8,0.9])
net.create_architecture()
saver = tf.train.Saver()
saver.restore(sess, tfmodel)
print('Loaded network {:s}'.format(tfmodel))

################################### one image demo #########################################
# image = cv2.imread('/home/oeasy/PycharmProjects/oeasy_face_lib_bak/example/oeasy_2.jpg')
# print('>>>>>image', image.shape)
# image_convert = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# im = image_convert[np.newaxis, :, :, :]
# total_boxes, points = net.test_image(sess, im)
# for i in range(total_boxes.shape[0]):
#     bbox = total_boxes[i]
#     corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
#     cv2.rectangle(image, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 2)
# for i in range(points.shape[0]):
#     for k in range(len(points[i]) // 2):
#         cv2.circle(image, (int(points[i][k]), int(int(points[i][k + 5]))), 3, (0, 0, 255))
# cv2.imshow('demo', image)
# cv2.waitKey(0)

################################### images demo #########################################
images_path = os.path.join(f_path, 'examples')
im_names = os.listdir(images_path)
print('>>>>', im_names)
for im_name in im_names:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Demo for {}/{}'.format(images_path, im_name))
    img = cv2.imread(os.path.join(images_path, im_name))
    print(img.shape)
    # image_convert = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = image_convert[np.newaxis, :, :, :]
    total_boxes, points = net.test_image(sess, im)
    for i in range(total_boxes.shape[0]):
        bbox = total_boxes[i]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 2)
    for i in range(points.shape[0]):
        for k in range(len(points[i]) // 2):
            cv2.circle(img, (int(points[i][k]), int(int(points[i][k + 5]))), 3, (0, 0, 255))
    cv2.imshow('demo', img)
    cv2.waitKey(0)

