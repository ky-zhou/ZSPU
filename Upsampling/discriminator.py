import tensorflow as tf
from Common import ops

class Discriminator(object):
    def __init__(self, opts,is_training, name="Discriminator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False
        self.start_number = 32

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            inputs = tf.expand_dims(inputs,axis=2)
            with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
                features_local = ops.mlp_conv(inputs, [self.start_number, self.start_number * 2]) #b, n/r, 1, 64
                features_global = tf.reduce_max(features_local, axis=1, keep_dims=True, name='maxpool_0')
                features = tf.concat([features_local, tf.tile(features_global, [1, tf.shape(inputs)[1],1, 1])], axis=-1) # b, n/r, 1, 128
                features = ops.attention_unit(features, is_training=self.is_training) #b, n/r, 1, 128
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
                features = ops.mlp_conv(features, [self.start_number * 4, self.start_number * 8]) #b, n/r, 1, 256
                features = tf.reduce_max(features, axis=1, name='maxpool_1') #b, 1, 256

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                outputs = ops.mlp(features, [self.start_number * 8, 1]) #b, 1, 1
                outputs = tf.reshape(outputs, [-1, 1]) #b, 1

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return outputs