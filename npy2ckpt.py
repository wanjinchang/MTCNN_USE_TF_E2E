#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: npy2ckpt.py
@time: 18-10-9 上午10:06
@desc:
'''
import tensorflow as tf
from detect_face_tf import Network
import numpy as np
import os
from six import string_types, iteritems

def save(saver, sess, logdir):
    model_name = 'mtcnn.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, write_meta_graph=True)
    print('The weights have been converted to {}.'.format(checkpoint_path))

def convert(output_path):
    ################################# Convert three npy of MTCNN to one ckpt #######################################
    net = Network()
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    scope = 'detect_face'
    ignore_missing = False

    with tf.Session(config=tfconfig) as sess:
        layers = net.create_architecture()
        variables = tf.global_variables()
        for var in variables:
            print("variables:", var)

        sess.run(tf.global_variables_initializer())
        with tf.variable_scope(scope, reuse=True):
            ###################################### pnet params  #######################################
            with tf.variable_scope('PNet', reuse=True):
                print('~~~~~~~~~~~~~~~~~~~~~Restoring PNet params~~~~~~~~~~~~~~~~~~~~~')
                data_dict = np.load('./mtcnn_model/det1.npy', encoding='latin1').item()  # pylint: disable=no-member

                for op_name in data_dict:
                    # print('>>>>>op_name', op_name)
                    for param_name, data in iteritems(data_dict[op_name]):
                        try:
                            if param_name == 'alpha' and op_name.startswith('PReLU'):
                                print('assigning %s' % op_name + '/' + param_name)
                                var = tf.get_variable('conv%s' % op_name[-1] + '/' + param_name)
                                sess.run(var.assign(data))
                            elif param_name == 'weights':
                                print('assigning %s' % op_name + '/' + param_name)
                                weights_var = tf.get_variable(op_name + '/' + param_name)
                                sess.run(weights_var.assign(data))
                            elif param_name == 'biases':
                                print('assigning %s' % op_name + '/' + param_name)
                                biases_var = tf.get_variable(op_name + '/' + param_name)
                                sess.run(biases_var.assign(data))
                        except ValueError:
                            if not ignore_missing:
                                raise

            ###################################### rnet params  #######################################
            with tf.variable_scope('RNet', reuse=True):
                print('~~~~~~~~~~~~~~~~~~~~~Restoring RNet params~~~~~~~~~~~~~~~~~~~~~')
                data_dict = np.load('./mtcnn_model/det2.npy', encoding='latin1').item()  # pylint: disable=no-member

                for op_name in data_dict:
                    # print('>>>>>op_name', op_name)
                    for param_name, data in iteritems(data_dict[op_name]):
                        try:
                            if param_name == 'alpha' and op_name.startswith('prelu'):
                                print('assigning %s' % op_name + '/' + param_name)
                                var = tf.get_variable('conv%s' % op_name[-1] + '/' + param_name)
                                sess.run(var.assign(data))
                            elif param_name == 'weights':
                                print('assigning %s' % op_name + '/' + param_name)
                                weights_var = tf.get_variable(op_name + '/' + param_name)
                                # weights_var = tf.get_variable(param_name)
                                sess.run(weights_var.assign(data))
                            elif param_name == 'biases':
                                print('assigning %s' % op_name + '/' + param_name)
                                biases_var = tf.get_variable(op_name + '/' + param_name)
                                sess.run(biases_var.assign(data))
                        except ValueError:
                            if not ignore_missing:
                                raise

            ###################################### onet params  #######################################
            with tf.variable_scope('ONet', reuse=True):
                print('~~~~~~~~~~~~~~~~~~~~~Restoring ONet params~~~~~~~~~~~~~~~~~~~~~')
                data_dict = np.load('./mtcnn_model/det3.npy', encoding='latin1').item()  # pylint: disable=no-member

                for op_name in data_dict:
                    # print('>>>>>op_name', op_name)
                    for param_name, data in iteritems(data_dict[op_name]):
                        try:
                            if param_name == 'alpha' and op_name.startswith('prelu'):
                                print('assigning %s' % op_name + '/' + param_name)
                                var = tf.get_variable('conv%s' % op_name[-1] + '/' + param_name)
                                sess.run(var.assign(data))
                            elif param_name == 'weights':
                                print('assigning %s' % op_name + '/' + param_name)
                                weights_var = tf.get_variable(op_name + '/' + param_name)
                                sess.run(weights_var.assign(data))
                            elif param_name == 'biases':
                                print('assigning %s' % op_name + '/' + param_name)
                                biases_var = tf.get_variable(op_name + '/' + param_name)
                                sess.run(biases_var.assign(data))
                        except ValueError:
                            if not ignore_missing:
                                raise

            saver = tf.train.Saver(var_list=variables, write_version=1)
            save(saver, sess, output_path)
            print("\nModel saved in file: %s" % output_path)

if __name__ == '__main__':
    output_path = '/mtcnn_model/mtcnn_ckpt/'
    convert(output_path)