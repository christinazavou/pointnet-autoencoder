""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.
Using GPU Chamfer's distance loss.

Author: Charles R. Qi
Date: May 2018
"""
import tensorflow as tf
# import tensorflow_graphics as tfg

from utils import tf_util
from utils import tfg
import numpy as np


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl


def get_debug_encoder(point_cloud, num_point, point_dim, is_training, bn_decay=None):
    """ Encoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """

    debug_checks = {}

    input_image = tf.expand_dims(point_cloud, -1)
    debug_checks['input_image'] = input_image

    # Encoder
    net = tf_util.conv2d(input_image, 64, [1, point_dim],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    debug_checks['conv1'] = net

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    debug_checks['conv2'] = net

    point_feat = tf_util.conv2d(net, 64, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv3', bn_decay=bn_decay)
    debug_checks['conv3'] = point_feat

    net = tf_util.conv2d(point_feat, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    debug_checks['conv4'] = net

    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    debug_checks['conv5'] = net

    global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                     padding='VALID', scope='maxpool')
    debug_checks['global_feat'] = global_feat

    net = tf.squeeze(tf.squeeze(global_feat, 1), 1)
    debug_checks['embedding'] = net

    mu = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc_mu', bn_decay=bn_decay)
    debug_checks['mu'] = mu

    logvar = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc_logvar', bn_decay=bn_decay)
    debug_checks['logvar'] = logvar

    return mu, logvar, debug_checks


def get_debug_decoder(net, num_point, is_training, bn_decay=None):
    debug_checks = {}
    # FC Decoder
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    debug_checks['fc1'] = net
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    debug_checks['fc2'] = net
    net = tf_util.fully_connected(net, num_point * 3, activation_fn=None, scope='fc3')
    debug_checks['fc3'] = net
    net = tf.reshape(net, (-1, num_point, 3))  # the new generated sample

    return net, debug_checks


def reparameterise(mu, logvar, is_training):
    # sample from normal distribution with mean mu and std logvar using the reparameterization trick
    zs = tf.cond(is_training,
                 lambda: tf.random.normal(shape=mu.shape) * tf.exp(logvar * .5) + mu,
                 lambda: mu)
    return zs


def get_debug_model(point_cloud, is_training, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    # batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value

    debug_checks = {}

    mu, logvar, dbg = get_debug_encoder(point_cloud, num_point, point_dim, is_training, bn_decay)
    debug_checks.update(dbg)

    zs = reparameterise(mu, logvar, is_training)

    dec, dbg = get_debug_decoder(zs, num_point, is_training, bn_decay)
    debug_checks.update(dbg)
    return zs, mu, logvar, dec, debug_checks


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def get_loss(z, mu, logvar, pred, label):
    """ z: Bx1x1024
        pred: BxNx3,
        label: BxNx3, """
    recon_loss = tf.reduce_mean(tfg.evaluate(pred, label))  # chamfer
    kl_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(logvar) + mu ** 2 - 1. - logvar, axis=1))

    # logpx_z = -tf.reduce_sum(chamfer_dist, axis=[1, 2])
    # logpz = log_normal_pdf(z, 0., 0.)
    # logqz_x = log_normal_pdf(z, mu, logvar)
    # return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    return recon_loss + kl_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.random.uniform((32, 2048, 3))
        zs, mu, logvar, dec, dbg = get_debug_model(inputs, True, bn_decay=None)

        loss = get_loss(zs, mu, logvar, dec, inputs)

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Trainable parameters: {}".format(total_parameters))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            zs_, mu_, logvar_, dec_, loss_ = sess.run([zs, mu, logvar, dec, loss])
            print(zs_.shape, dec_.shape, loss_.shape)
            debug_checks = sess.run(dbg)
            for key, value in debug_checks.items():
                print("{}: {}".format(key, value.shape))
