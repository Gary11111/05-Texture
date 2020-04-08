# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

import vgg16 as vgg16
from common import config

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


'''use 13 convolution layers to generate gram matrix'''
GRAM_LAYERS = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
               'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

image_shape = (1, 224, 224, 3)

'''you need to complete this method'''


def get_l2_loss(noise, source):  # 获得l2-loss
    l2_loss = tf.reduce_sum((source - noise) ** 2)
    return l2_loss


def get_tv_loss(img):
    left_loss = tf.reduce_sum((img[1:, :, :] - img[-1:, :, :]) ** 2)
    down_loss = tf.reduce_sum((img[:, 1:, :] - img[:, -1:, :]) ** 2)
    tv_loss = left_loss + down_loss
    # print("tv loss ===> ", tv_loss)
    return tv_loss


def gram_matrix(feature_map, normalize=True):  # 获得gram矩阵
    shape = tf.shape(feature_map)
    """三维矩阵相乘的技巧"""
    features_reshaped = tf.reshape(feature_map, (shape[1] * shape[2], shape[3]))  # 展成二维
    gram = tf.matmul(tf.transpose(features_reshaped), features_reshaped)  # 乘以转置得到channel*channel的矩阵
    if normalize:
        gram /= tf.cast((shape[3] * shape[1] * shape[2]), tf.float32)  # 标准化
    return gram


def get_gram_loss(noise_layer, source_layer):
    """layer中包含所有的feature map each layer's shape: (batch,w,h,channel)"""
    noise_gram = gram_matrix(noise_layer)
    source_gram = gram_matrix(source_layer)  # gram矩阵的维度为channel*channel
    gram_loss = tf.reduce_sum((source_gram - noise_gram) ** 2)  # 得到一个value
    # print("gram_loss === > ", gram_loss)
    return gram_loss


def get_EMD_loss(noise_layer, source_layer):
    shape = tf.shape(noise_layer)
    """三维矩阵相乘的技巧"""
    noise_reshaped = tf.reshape(noise_layer, (shape[1] * shape[2], shape[3]))  # 展成二维: 每一行是一个feature map
    source_reshaped = tf.reshape(source_layer, (shape[1] * shape[2], shape[3]))
    noise_sort = tf.sort(noise_reshaped, direction='ASCENDING')
    source_sort = tf.sort(source_reshaped, direction='ASCENDING')
    return tf.reduce_sum(tf.math.square(noise_sort - source_sort))


def get_l2_gram_loss_for_layer(noise, source, layer, weight=1):
    noise_layer = getattr(noise, layer)
    source_layer = getattr(source, layer)
    l2_loss = get_l2_loss(noise_layer, source_layer)  # 得到l2loss
    tv_loss = get_tv_loss(noise_layer)
    gram_loss = get_gram_loss(noise_layer, source_layer)
    emd_loss = get_EMD_loss(noise_layer, source_layer)
    l2_loss_gram = weight * emd_loss + 0 * gram_loss + 0 * l2_loss + 0.00 * tv_loss
    return l2_loss_gram


def get_loss(noise, source):
    total_loss = []
    for layer in GRAM_LAYERS:
        total_loss.append(get_l2_gram_loss_for_layer(noise, source, layer))
    return tf.reduce_mean(tf.convert_to_tensor(total_loss))


def output_img(sess, inp, save=False, out_path=None):
    shape = image_shape
    img = np.clip(sess.run(inp), 0, 1) * 255
    img = img.astype('uint8')
    if save:
        cv2.imwrite(out_path, (np.reshape(img, shape[1:])))


def main():
    tf.disable_eager_execution()
    '''training a image initialized with noise'''
    noise = tf.Variable(tf.nn.sigmoid(tf.random_uniform(image_shape, -3, 3)))
    '''load texture image, notice that the pixel value has to be normalized to [0,1]'''
    image = cv2.imread('./images/red-peppers256.jpg')
    image = cv2.resize(image, image_shape[1:3])
    image = image.reshape(image_shape)
    image = (image / 255).astype('float32')

    ''' get features of the texture image and the generated image'''
    image_model = vgg16.Vgg16()
    image_model.build(image)
    noise_model = vgg16.Vgg16()
    noise_model.build(noise)

    loss = get_loss(noise_model, image_model)

    global_steps = tf.Variable(0, trainable=False)
    values = [0.01, 0.005, 0.001]
    boundaries = [200, 1500]
    lr = tf.train.piecewise_constant(global_steps, boundaries=boundaries, values=values)  # 调整学习率的方法

    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss, [noise])
    update_image = opt.apply_gradients(grads)  # SGD优化noise: 更新的noise

    ''' create a session '''
    tf.set_random_seed(12345)  # ensure consistent results
    global_cnt = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.nr_epoch):
            global_cnt += 1
            ''' compute loss based on gram matrix'''
            _, _loss, _lr = sess.run([update_image, loss, lr], feed_dict={global_steps: epoch})
            if global_cnt % config.show_interval == 0:
                print("e:{}".format(epoch), "loss:{}".format(_loss), "lr:", _lr)
            '''save the trained image every 100 epoch'''
            if global_cnt % config.save_interval == 0 and global_cnt > 0:
                out_dir = './out'
                out_dir = out_dir + '/{}.png'.format(global_cnt)
                output_img(sess, noise, save=True, out_path=out_dir)
        print('Training is done, exit.')


if __name__ == "__main__":
    main()
