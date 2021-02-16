""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.
Using GPU Chamfer's distance loss.

Author: Charles R. Qi
Date: May 2018
"""
import tensorflow as tf
# import tensorflow_graphics as tfg

from utils import tf_util
from utils import tfg


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    # Encoder
    net = tf_util.conv2d(input_image, 64, [1, point_dim],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                     padding='VALID', scope='maxpool')

    net = tf.reshape(global_feat, [batch_size, -1])
    end_points['embedding'] = net

    # FC Decoder
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_point * 3, activation_fn=None, scope='fc3')
    net = tf.reshape(net, (batch_size, num_point, 3))

    return net, end_points


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

    debug_checks = {}

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    end_points = {}

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

    net = tf.reshape(global_feat, [batch_size, -1])
    end_points['embedding'] = net
    debug_checks['embedding'] = net

    # FC Decoder
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    debug_checks['fc1'] = net

    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    debug_checks['fc2'] = net

    net = tf_util.fully_connected(net, num_point * 3, activation_fn=None, scope='fc3')
    debug_checks['fc3'] = net

    net = tf.reshape(net, (batch_size, num_point, 3))
    debug_checks['prediction'] = net

    return net, end_points, debug_checks


def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    loss = tf.reduce_mean(tfg.evaluate(pred, label))
    end_points['pcloss'] = loss
    return loss * 100, end_points


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.random.uniform((32, 1024, 3))
        outputs = get_debug_model(inputs, tf.constant(True))
        loss = get_loss(outputs[0], inputs, outputs[1])
        print(loss)

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
            predictions, embeddings, loss = sess.run([outputs[0], outputs[1]['embedding'], loss[0]])
            print(predictions.shape, embeddings.shape, loss.shape)
            debug_checks = sess.run(outputs[2])
            for key, value in debug_checks.items():
                print("{}: {}".format(key, value.shape))
