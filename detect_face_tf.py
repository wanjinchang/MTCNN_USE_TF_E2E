""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib import slim
import tensorflow as tf
#from math import floor
import cv2

from boxes_utils import bbreg_tf, rerec_tf, pad_tf, generateBoundingBox_tf

class Network(object):
    def __init__(self, minsize=100, threshold=[0.6,0.8,0.95], factor=0.709):
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor

    def prelu(self, inp):
        # i = int(inp.get_shape()[-1])
        # alpha = self.make_var('alpha', shape=(i,))
        alpha = tf.get_variable("alpha", shape=inp.get_shape()[-1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.25))
        output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    def P_Net(self, inputs):
        # define common param
        with tf.variable_scope('PNet', reuse=None):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=self.prelu,
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='valid'):
                print(inputs.get_shape())
                net = slim.conv2d(inputs, 10, [3, 3], stride=1, scope='conv1')
                print(net.get_shape())
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1', padding='SAME')
                print(net.get_shape())
                net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
                print(net.get_shape())
                net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='conv3')
                print(net.get_shape())
                # batch*H*W*2
                conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv4-1',
                                      activation_fn=tf.nn.softmax)
                print(conv4_1.get_shape())
                # batch*H*W*4
                bbox_pred = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv4-2',
                                        activation_fn=None)
                print(bbox_pred.get_shape())
                # cls_prob_original = conv4_1
                # bbox_pred_original = bbox_pred
                # when test,batch_size = 1
                cls_prob = conv4_1
                # cls_prob = tf.squeeze(conv4_1, axis=0)
                # bbox_pred = tf.squeeze(bbox_pred, axis=0)
                return cls_prob, bbox_pred

    def R_Net(self, inputs):
        with tf.variable_scope('RNet', reuse=None):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=self.prelu,
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='valid'):
                print(inputs.get_shape())
                net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3, 3], stride=1, scope="conv1")
                print(net.get_shape())
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
                print(net.get_shape())
                net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=1, scope="conv2")
                print(net.get_shape())
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
                print(net.get_shape())
                net = slim.conv2d(net, num_outputs=64, kernel_size=[2, 2], stride=1, scope="conv3")
                print(net.get_shape())
                fc_flatten = slim.flatten(net)
                print(fc_flatten.get_shape())
                fc1 = slim.fully_connected(fc_flatten, num_outputs=128, scope="conv4", activation_fn=self.prelu)
                print(fc1.get_shape())
                # batch*2
                cls_prob = slim.fully_connected(fc1, num_outputs=2, scope="conv5-1", activation_fn=tf.nn.softmax)
                print(cls_prob.get_shape())
                # batch*4
                bbox_pred = slim.fully_connected(fc1, num_outputs=4, scope="conv5-2", activation_fn=None)
                print(bbox_pred.get_shape())
                return cls_prob, bbox_pred

    def O_Net(self, inputs):
        with tf.variable_scope('ONet', reuse=None):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=self.prelu,
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='valid'):
                print(inputs.get_shape())
                net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3, 3], stride=1, scope="conv1")
                print(net.get_shape())
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
                print(net.get_shape())
                net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv2")
                print(net.get_shape())
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
                print(net.get_shape())
                net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv3")
                print(net.get_shape())
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
                print(net.get_shape())
                net = slim.conv2d(net, num_outputs=128, kernel_size=[2, 2], stride=1, scope="conv4")
                # print(net.get_shape())
                fc_flatten = slim.flatten(net)
                print(fc_flatten.get_shape())
                fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope="conv5", activation_fn=self.prelu)
                print(fc1.get_shape())
                # batch*2
                cls_prob = slim.fully_connected(fc1, num_outputs=2, scope="conv6-1", activation_fn=tf.nn.softmax)
                print(cls_prob.get_shape())
                # batch*4
                bbox_pred = slim.fully_connected(fc1, num_outputs=4, scope="conv6-2", activation_fn=None)
                print(bbox_pred.get_shape())
                # batch*10
                landmark_pred = slim.fully_connected(fc1, num_outputs=10, scope="conv6-3", activation_fn=None)
                print(landmark_pred.get_shape())
                return cls_prob, bbox_pred, landmark_pred

    def detect_face_tf(self):
        with tf.variable_scope('detect_face', reuse=None):
            factor_count = tf.constant(0.0, dtype=tf.float32)
            h = tf.to_float(tf.shape(self._image)[1])
            w = tf.to_float(tf.shape(self._image)[2])
            minl = tf.minimum(h, w)
            m = tf.div(tf.constant(12, dtype=tf.float32), tf.constant(self.minsize, dtype=tf.float32))
            minl = minl * m

            pnet_size = tf.constant(12.0, dtype=tf.float32)
            def condition(i, minl, scales, factor_count):
                r = tf.greater_equal(minl, pnet_size)
                return r
            def body(i, minl, scales, factor_count):
                scale = m * tf.pow(self.factor, factor_count)
                # scales = tf.concat([scales, [scale]], 0)
                scales = scales.write(i, scale)
                factor_count += 1.0
                return i+1, minl*self.factor, scales, factor_count

            i = tf.constant(0, dtype=tf.int32)
            scales = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1,
                                         dynamic_size=True, clear_after_read=True)
            # generate scales
            [i, minl, scales, factor_count] = tf.while_loop(condition, body, [i, minl, scales, factor_count])
            scales = scales.stack()

            # first stage --> pnet process
            def cond1(i, total_boxes):
                r = tf.less(i, tf.shape(scales)[0])
                return r
            def body1(i, total_boxes):
                scale = scales[i]
                hs = tf.to_int32(tf.ceil(h * scale))
                ws = tf.to_int32(tf.ceil(w * scale))
                img_tensor = self._image[0, :, :, :]
                im_data = tf.py_func(resample_np, [img_tensor, (hs, ws)], tf.float32)
                im_data.set_shape([None, None, None, 3])
                img_y = tf.transpose(im_data, perm=[0, 2, 1, 3])
                cls_prob, bbox_pred = self.P_Net(img_y)
                cls_prob = tf.transpose(cls_prob, perm=[0, 2, 1, 3])
                bbox_pred = tf.transpose(bbox_pred, perm=[0, 2, 1, 3])
                # generate bounding boxes for each scale
                boxes, _ = generateBoundingBox_tf(cls_prob[0, :, :, 1], bbox_pred[0, :, :, :], scale,
                                                         self.threshold[0])
                indices = tf.image.non_max_suppression(boxes[:, 0:4], boxes[:, 4], max_output_size=10000,
                                                       iou_threshold=0.5)
                boxes = tf.gather(boxes, indices)
                boxes = tf.to_float(boxes)
                total_boxes = total_boxes.write(i, boxes)
                return i+1, total_boxes

            i = tf.constant(0, dtype=tf.int32)
            total_boxes = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1,
                                      dynamic_size=True, clear_after_read=True)
            [i, total_boxes] = tf.while_loop(cond1, body1, [i, total_boxes])

            # generate and refine the result and bounding boxes of pnet
            total_boxes = total_boxes.concat()
            result, total_boxes = tf.cond(tf.greater(tf.shape(total_boxes)[0], 0), lambda: self.refine_boxes_pnet(total_boxes, h, w),
                             lambda: (tf.constant([], shape=[0, 10], dtype=tf.int32), tf.constant([], shape=[0, 5], dtype=tf.float32)))
            result.set_shape([None, 10])
            zeros_tensor = tf.zeros(shape=(1, 24, 24, 3), dtype=tf.float32)

            # second stage --> rnet process
            # get the input data result from the pnet result
            im_data = tf.cond(tf.greater(tf.shape(total_boxes)[0], 0), lambda: self.rnet_process(total_boxes, result),
                              lambda: zeros_tensor)
            cls_prob, bbox_pred = self.R_Net(im_data)
            total_boxes = self.rnet_select(cls_prob, bbox_pred, total_boxes)

            # third stage --> onet process
            # get the input data result from the rnet result
            zeros_tensor = tf.zeros(shape=(1, 48, 48, 3), dtype=tf.float32)
            im_data = tf.cond(tf.greater(tf.shape(total_boxes)[0], 0),
                              lambda: self.onet_process(total_boxes, w, h),
                              lambda: zeros_tensor)
            cls_prob, bbox_pred, landmark_pred = self.O_Net(im_data)
            self.cls_prob, self.bbox_pred, self.landmark_pred = cls_prob, bbox_pred, landmark_pred
            self.total_boxes, self.points = self.onet_select(cls_prob, bbox_pred, landmark_pred, total_boxes)
            all_boxes = tf.identity(self.total_boxes, name='total_boxes')
            all_points = tf.identity(self.points, name='points')
            return all_boxes, all_points

    def refine_boxes_pnet(self, total_boxes, h, w):
        """
        nms and get the offset of pnet.
        :param total_boxes: tensor of size (None, 9).
        :param h: the height of the input image.
        :param w: the weight of the input image.
        :return: the offset result and refined boxes.
        """
        pick = tf.image.non_max_suppression(total_boxes[:, 0:4], total_boxes[:, 4], max_output_size=1000,
                                            iou_threshold=0.7)
        total_boxes = tf.gather(total_boxes, pick)
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = tf.stack([qq1, qq2, qq3, qq4, total_boxes[:, 4]], axis=1)
        total_boxes = rerec_tf(total_boxes)
        total_boxes = tf.floor(total_boxes)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad_tf(total_boxes, w, h)
        result = tf.stack([dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph], axis=1)
        return result, total_boxes

    def rnet_process(self, total_boxes, result):
        """
        get the input data of rnet according to the offset result.
        """
        zeros_tensor = tf.zeros(shape=(1, 24, 24, 3), dtype=tf.float32)

        def cond2(i, tempimgs):
            r = tf.less(i, tf.shape(total_boxes)[0])
            return r

        def body2(i, tempimgs):
            img_tensor = self._image[0, result[i, 4]-1:result[i, 5], result[i, 6]-1:result[i, 7], :]
            pred = (tf.greater(tf.shape(img_tensor)[0], 0)) & (tf.greater(tf.shape(img_tensor)[1], 0))
            tempimg = tf.cond(pred, lambda: tf.py_func(resample_np, [img_tensor, (24, 24)], tf.float32), lambda: zeros_tensor)
            tempimgs = tempimgs.write(i, tempimg)
            return i+1, tempimgs

        i = tf.constant(0, dtype=tf.int32)
        tempimgs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1,
                                  dynamic_size=True, clear_after_read=True)
        [i, tempimgs] = tf.while_loop(cond2, body2, [i, tempimgs])
        tempimgs = tempimgs.concat()
        tempimgs.set_shape([None, 24, 24, 3])
        im_data = tf.transpose(tempimgs, perm=[0, 2, 1, 3])
        return im_data

    def onet_process(self, total_boxes, w, h):
        """
        get the input data of onet according to the offset result.
        """
        zeros_tensor = tf.zeros(shape=(1, 48, 48, 3), dtype=tf.float32)

        total_boxes = tf.to_int32(tf.floor(total_boxes))
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad_tf(total_boxes, w, h)
        result = tf.stack([dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph], axis=1)

        def cond2(i, tempimgs):
            r = tf.less(i, tf.shape(total_boxes)[0])
            return r

        def body2(i, tempimgs):
            img_tensor = self._image[0, result[i, 4]-1:result[i, 5], result[i, 6]-1:result[i, 7], :]
            pred = (tf.greater(tf.shape(img_tensor)[0], 0)) & (tf.greater(tf.shape(img_tensor)[1], 0))
            tempimg = tf.cond(pred, lambda: tf.py_func(resample_np, [img_tensor, (48, 48)], tf.float32), lambda: zeros_tensor)
            tempimgs = tempimgs.write(i, tempimg)
            return i+1, tempimgs

        i = tf.constant(0, dtype=tf.int32)
        tempimgs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1,
                                  dynamic_size=True, clear_after_read=True)
        [i, tempimgs] = tf.while_loop(cond2, body2, [i, tempimgs])
        tempimgs = tempimgs.concat()
        tempimgs.set_shape([None, 48, 48, 3])
        im_data = tf.transpose(tempimgs, perm=[0, 2, 1, 3])
        return im_data

    def rnet_select(self, cls_prob, bbox_pred, total_boxes):
        """
        select the bounding boxes and scores that over threshold.
        :param cls_prob: tensor of size (None, 2).
        :param bbox_pred: tensor of size (None, 4).
        :param total_boxes: tensor of size (None, 5).
        :return: refine boxes of size (None, 5).
        """
        score = cls_prob[:, 1]
        ipass = tf.where(tf.greater(score, self.threshold[1]))
        score_pass = tf.gather_nd(score, ipass)
        boxes_pass = tf.gather_nd(total_boxes[:, 0:4], ipass)
        total_boxes = tf.concat([boxes_pass[:, 0:4], tf.expand_dims(score_pass, 1)], axis=1)
        mv = tf.gather_nd(bbox_pred, ipass)
        boxes_empty = tf.constant([], shape=(0, 5), dtype=tf.float32)
        total_boxes = tf.cond(tf.greater(tf.shape(total_boxes)[0], 0), lambda: self.rnet_boxes(total_boxes, mv),
                              lambda: boxes_empty)
        return total_boxes

    def onet_select(self, cls_prob, bbox_pred, landmark_pred, total_boxes):
        """
        select the bounding boxes, scores and landmarks that over threshold.
        :param cls_prob: tensor of size (None, 2).
        :param bbox_pred: tensor of size (None, 4).
        :param landmark_pred: tensor of size (None, 10).
        :param total_boxes: tensor of size (None, 5).
        :return: refined boxes and landmarks.
        """
        score = cls_prob[:, 1]
        ipass = tf.where(tf.greater(score, self.threshold[2]))
        score_pass = tf.gather_nd(score, ipass)
        points = tf.gather_nd(landmark_pred, ipass)
        boxes_pass = tf.gather_nd(total_boxes, ipass)
        mv = tf.gather_nd(bbox_pred, ipass)
        total_boxes = tf.concat([boxes_pass[:, 0:4], tf.expand_dims(score_pass, 1)], axis=1)

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        tile_w = tf.tile(tf.expand_dims(w, axis=1), [1, 5])
        tile_h = tf.tile(tf.expand_dims(h, axis=1), [1, 5])
        boxes_x_tile = tf.tile(tf.expand_dims(total_boxes[:, 0], axis=1), [1, 5])
        boxes_y_tile = tf.tile(tf.expand_dims(total_boxes[:, 1], axis=1), [1, 5])
        points_list = []
        points_list.append(tile_w * points[:, 0:5] + boxes_x_tile - 1)
        points_list.append(tile_h * points[:, 5:10] + boxes_y_tile - 1)
        points = tf.concat(points_list, axis=1)

        boxes_empty = tf.constant([], shape=(0, 5), dtype=tf.float32)
        points_empty = tf.constant([], shape=(0, 10), dtype=tf.float32)
        total_boxes, points = tf.cond(tf.greater(tf.shape(total_boxes)[0], 0),
                                      lambda: self.onet_boxes(total_boxes, mv, points),
                                      lambda: (boxes_empty, points_empty))
        return total_boxes, points

    def rnet_boxes(self, total_boxes, mv):
        """
        nms and refined boxes.
        """
        pick = tf.image.non_max_suppression(total_boxes[:, 0:4], total_boxes[:, 4], max_output_size=1000,
                                            iou_threshold=0.5)
        total_boxes = tf.gather(total_boxes, pick)
        reg = tf.gather(mv, pick)
        total_boxes = bbreg_tf(total_boxes, reg)
        total_boxes = rerec_tf(total_boxes)
        return total_boxes

    def onet_boxes(self, total_boxes, mv, points):
        """
        nms and refined boxes.
        """
        total_boxes = bbreg_tf(total_boxes, mv)
        pick = tf.image.non_max_suppression(total_boxes[:, 0:4], total_boxes[:, 4], max_output_size=1000,
                                            iou_threshold=0.5)
        total_boxes = tf.gather(total_boxes, pick)
        points = tf.gather(points, pick)
        return total_boxes, points

    def create_architecture(self):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image_tensor')

        layers_to_output = {}
        total_boxes, points = self.detect_face_tf()
        layers_to_output['boxes'] = total_boxes
        layers_to_output['points'] = points
        return layers_to_output

    def test_image(self, sess, image):
        feed_dict = {self._image: image}
        total_boxes, points = sess.run([self.total_boxes, self.points], feed_dict=feed_dict)
        return total_boxes, points

def resample_np(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    im_data = (im_data - 127.5) * 0.0078125
    im_data = np.expand_dims(im_data, axis=0)
    return im_data

def imresample_tf(img, sz):
    img_data = tf.image.resize_bilinear(img, (sz[0], sz[1]), name='downsampling', align_corners=True)
    return img_data


