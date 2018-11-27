#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: boxes_utils.py
@time: 18-10-8 下午2:38
@desc:
'''
import tensorflow as tf

import cv2
import numpy as np

def bbreg_tf(boundingbox, reg):
    # calibrate bounding boxes
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    score = boundingbox[:, 4]
    boundingbox = tf.stack([b1, b2, b3, b4, score], axis=1)
    return boundingbox

def generateBoundingBox_tf(imap, reg, scale, t):
    # use tf to generate heatmap to bounding boxes
    stride = 2.0
    cellsize = 12.0
    imap = tf.transpose(imap, perm=[1, 0])

    dx1 = tf.transpose(reg[:, :, 0], perm=[1, 0])
    dy1 = tf.transpose(reg[:, :, 1], perm=[1, 0])
    dx2 = tf.transpose(reg[:, :, 2], perm=[1, 0])
    dy2 = tf.transpose(reg[:, :, 3], perm=[1, 0])
    index = tf.where(tf.greater_equal(imap, t))
    def flipud(x1, y1, x2, y2):
        x1 = x1[::-1, ...]
        y1 = y1[::-1, ...]
        x2 = x2[::-1, ...]
        y2 = y2[::-1, ...]
        return x1, y1, x2, y2
    def no_flipud(x1, y1, x2, y2):
        return x1, y1, x2, y2
    dx1, dy1, dx2, dy2 = tf.cond(tf.equal(tf.shape(index)[0], 1), lambda: flipud(dx1, dy1, dx2, dy2),
                                          lambda: no_flipud(dx1, dy1, dx2, dy2))
    score = tf.gather_nd(imap, index)
    x1 = tf.gather_nd(dx1, index)
    y1 = tf.gather_nd(dy1, index)
    x2 = tf.gather_nd(dx2, index)
    y2 = tf.gather_nd(dy2, index)
    reg = tf.stack([x1, y1, x2, y2], axis=1)

    reg = tf.cond(tf.equal(tf.shape(reg)[0], 0), lambda: tf.constant([], shape=[0, 4]), lambda: reg)

    q1 = tf.to_float(tf.floor((stride * tf.cast(index, dtype=tf.float32) + 1.0) / scale))
    q2 = tf.to_float(tf.floor((stride * tf.cast(index, dtype=tf.float32) + cellsize - 1.0 + 1.0) / scale))

    boundingbox = tf.concat([q1, q2, tf.expand_dims(score, axis=1), reg], axis=1)
    return boundingbox, reg

def pad_tf(total_boxes, w, h):
    # compute the padding coordinates (pad the bounding boxes to square)
    tmpw = tf.to_int32(total_boxes[:, 2] - total_boxes[:, 0] + 1)
    tmph = tf.to_int32(total_boxes[:, 3] - total_boxes[:, 1] + 1)
    numbox = tf.shape(total_boxes)[0]
    w = tf.to_int32(w)
    h = tf.to_int32(h)

    w = tf.tile([w], tf.reshape(numbox, [1]))
    h = tf.tile([h], tf.reshape(numbox, [1]))

    dx = tf.ones([numbox], dtype=tf.int32)
    dy = tf.ones([numbox], dtype=tf.int32)
    edx = tmpw
    edy = tmph

    x = tf.to_int32(total_boxes[:, 0])
    y = tf.to_int32(total_boxes[:, 1])
    ex = tf.to_int32(total_boxes[:, 2])
    ey = tf.to_int32(total_boxes[:, 3])

    tmp_index = tf.greater(ex, w)
    edx = tf.where(tmp_index, -ex + w + tmpw, edx)
    ex = tf.where(tmp_index, w, ex)

    tmp_index = tf.greater(ey, h)
    edy = tf.where(tmp_index, -ey + h + tmph, edy)
    ey = tf.where(tmp_index, h, ey)

    tmp_index = tf.less(x, 1)
    dx = tf.where(tmp_index, 2 - x, dx)
    x = tf.where(tmp_index, tf.ones_like(x), x)

    tmp_index = tf.less(y, 1)
    dy = tf.where(tmp_index, 2 - y, dy)
    y = tf.where(tmp_index, tf.ones_like(y), y)

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

def rerec_tf(bboxA):
    # convert bboxA to square
    bbox = []
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = tf.maximum(w, h)
    bbox.append(bboxA[:, 0] + w * 0.5 - l * 0.5)
    bbox.append(bboxA[:, 1] + h * 0.5 - l * 0.5)
    bbox.append(bboxA[:, 0] + w * 0.5 + l * 0.5)
    bbox.append(bboxA[:, 1] + h * 0.5 + l * 0.5)
    bbox.append(bboxA[:, 4])
    bboxes = tf.stack(bbox, axis=1)
    return bboxes
