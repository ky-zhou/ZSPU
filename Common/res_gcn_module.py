import tensorflow as tf
from tf_ops.grouping.tf_grouping import group_point, knn_point
from Common.model_utils import gen_grid
from Common.ops import mlp_conv, conv2d


def group(xyz, points, k, dilation=1, use_xyz=False):
    """
    param xyz: b, n, 3
    param points: b, n, c
    return: b, n, k, 3;  b, n, k, c;   b, n, k
    """
    _, idx = knn_point(k*dilation+1, xyz, xyz)
    idx = idx[:, :, 1::dilation]

    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, k, 3)
    grouped_xyz -= tf.expand_dims(xyz, 2)  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, k, channel)
        if use_xyz:
            grouped_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, k, 3+channel)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


def gcn(xyz, k, n_cout, n_blocks, scope, activation=tf.nn.relu):
    """
    return: b, n, k, 3
    """
    with tf.variable_scope(scope):
        _, grouped_points, _ = group(xyz, None, k)      # b, n, k, 3
        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                grouped_points = conv2d(grouped_points, n_cout, (1, 1), scope='conv_xyz', padding='VALID',
                                        activation_fn=None, use_bias=False)
                if idx == n_blocks - 1:
                    return tf.reduce_max(grouped_points, axis=2)
                else:
                    grouped_points = activation(grouped_points)


def grid_module(points, n_cout, up_ratio=2):
    """
    Input: b, n, 1, c
    """
    batch_size, npoint = points.shape[0], points.shape[1]
    with tf.variable_scope('grid'):
        grid = gen_grid(up_ratio)       # 2, 2
        grid0 = tf.reshape(grid, [1, 1, -1])
        grid1 = tf.tile(tf.tile(grid0, [1, npoint, 1]), [batch_size, 1, 1])     # b, n, 1, 4
        points = tf.concat([points, grid1], axis=-1)        # b, n, 132
        points = mlp_conv(points, [n_cout])     # b, n, 1, c
        points = tf.nn.relu(points)
        return points


# def fps():


def res_gcn_up(xyz, points, k, n_cout, i_block, scope, indices=None, up_ratio=2, dim=3):
    """
    param xyz: b, n, 3
    param points: b, n, c
    return: b, n, 3;   b, n, c
    """
    with tf.variable_scope(scope):
        # Neighbor Features
        if indices is None:
            _, grouped_points, indices = group(xyz, points, k)
        else:
            grouped_points = group_point(points, indices)
        # Center Conv
        center_points = tf.expand_dims(points, axis=2)      #
        points = conv2d(center_points, n_cout, (1, 1), scope='conv_center', padding='VALID', use_bias=False)     # b, n, 1, c
        # Neighbor Conv
        grouped_points_nn = conv2d(grouped_points, n_cout, (1, 1), scope='conv_neighbor', padding='VALID',
                                   activation_fn=None, use_bias=False)     # b, n, k, c

        # Center Conv
        points_xyz = conv2d(points, dim*up_ratio, (1, 1), scope='conv_center_xyz', padding='VALID', use_bias=False)
        # Neighbor Conv
        grouped_points_xyz = conv2d(grouped_points_nn, dim*up_ratio, (1, 1), scope='conv_neighbor_xyz', padding='VALID',
                                    activation_fn=None, use_bias=False)

        new_xyz = tf.concat([points_xyz, grouped_points_xyz], axis=2)       # b, n, k+1, c
        new_xyz = tf.reduce_mean(new_xyz, axis=2, keepdims=True)
        """reshape module"""
        new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value, up_ratio, dim])
        new_xyz = new_xyz + tf.expand_dims(xyz, axis=2)
        new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value*up_ratio, dim])
        center_neighbor = gcn(new_xyz, 12, 128, 3, 'module_%d' % i_block)
        return new_xyz, center_neighbor
