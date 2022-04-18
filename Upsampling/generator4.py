import tensorflow as tf
from Common.res_gcn_module import *
import math


class Generator(object):
    def __init__(self, opts,is_training, reuse, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = reuse
        self.up_ratio = self.opts.up_ratio

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            block_num = int(math.log2(self.up_ratio))
            xyz = inputs[:, :, :self.opts.point_dim]
            points = gcn(xyz, 8, 128, 3, 'module_0')
            # b, n, k, 3
            for i in range(block_num):
                new_xyz, points = res_gcn_up(xyz, points, 8, 128, i, 'module_{}'.format(i+1),
                                             up_ratio=2, dim=self.opts.point_dim)
                xyz = new_xyz       # b, opt.num_point * opt.u, 3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return xyz
