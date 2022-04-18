import tensorflow as tf
from Upsampling.generator import Generator
from Upsampling.discriminator import Discriminator
from Common.visu_utils import plot_pcd_three_views, point_cloud_three_views
from Common.ops import add_scalar_summary, add_hist_summary
from Upsampling.data_loader import load_data
from Common import model_utils
from Common.loss_utils import *
import logging
import os
from time import time
from termcolor import colored
import numpy as np


class Model(object):
    def __init__(self, opts, sess, n_pcd, input, gt):
        self.sess = sess
        self.opts = opts
        self.opts.num_point = n_pcd//self.opts.up_ratio
        self.opts.n_pcd = n_pcd
        self.input, self.gt = input, gt

    def allocate_placeholders(self):
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.input_x = tf.placeholder(tf.float32, shape=[self.opts.batch_size, self.opts.num_point, self.opts.point_dim])
        self.input_y = tf.placeholder(tf.float32,
                                      shape=[self.opts.batch_size, self.opts.n_pcd, self.opts.point_dim])
        self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])

    def build_model(self):
        self.G = Generator(self.opts, self.is_training, False, name='generator')
        self.D = Discriminator(self.opts, self.is_training, name='discriminator')

        # X -> Y
        self.G_y = self.G(self.input_x)
        self.dist_loss = self.opts.fidelity_w * pc_distance(self.G_y[..., :3], self.input_y[..., :3], self.opts.dis_type, radius=self.pc_radius)
        # self.dist_loss = self.opts.fidelity_w * pc_distance(self.G_y[..., :3], self.input_x, self.opts.dis_type, radius=self.pc_radius)

        if self.opts.use_norm:
            self.norm_loss = self.opts.uniform_w * get_norm_loss(self.G_y[..., 3:])
        else:
            self.norm_loss = 0
        if self.opts.use_repulse:
            self.repulsion_loss = self.opts.repulsion_w * get_repulsion_loss(self.G_y[..., :3])
        else:
            self.repulsion_loss = 0
        self.uniform_loss = self.opts.uniform_w * get_uniform_loss(self.G_y[..., :3])
        self.pu_loss = self.dist_loss + self.repulsion_loss + self.uniform_loss +\
                       tf.losses.get_regularization_loss()
        # self.pu_loss = self.dist_loss + self.repulsion_loss + self.uniform_loss# + tf.losses.get_regularization_loss()

        if self.opts.use_gan:
            self.G_gan_loss = self.opts.gan_w * generator_loss(self.D, self.G_y)
        else:
            self.G_gan_loss = 0
        self.total_gen_loss = self.G_gan_loss + self.pu_loss
        if self.opts.use_gan:
            self.D_loss = discriminator_loss(self.D, self.input_y, self.G_y)
            # self.D_loss = discriminator_loss(self.D, self.input_x, self.G_y)

        self.setup_optimizer()
        self.summary_all()

        self.visualize_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
        self.visualize_titles = ['input_x', 'fake_y', 'real_y']

    def summary_all(self):

        # summary
        add_scalar_summary('loss/dis_loss', self.dist_loss, collection='gen')
        add_scalar_summary('loss/repulsion_loss', self.repulsion_loss, collection='gen')
        add_scalar_summary('loss/norm_loss', self.norm_loss, collection='gen')
        add_scalar_summary('loss/uniform_loss', self.uniform_loss, collection='gen')
        add_scalar_summary('loss/G_loss', self.G_gan_loss, collection='gen')
        add_scalar_summary('loss/total_gen_loss', self.total_gen_loss, collection='gen')
        if self.opts.use_gan:
            add_hist_summary('D/true', self.D(self.input_y), collection='dis')
            add_hist_summary('D/fake', self.D(self.G_y), collection='dis')
            add_scalar_summary('loss/D_Y', self.D_loss, collection='dis')

        self.g_summary_op = tf.summary.merge_all('gen')
        if self.opts.use_gan:
            self.d_summary_op = tf.summary.merge_all('dis')

        self.visualize_x_titles = ['input_x', 'fake_y', 'real_y']
        self.visualize_x_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
        self.image_x_merged = tf.placeholder(tf.float32, shape=[None, 1500, 1500, 1])
        self.image_x_summary = tf.summary.image('Upsampling', self.image_x_merged, max_outputs=1)

    def setup_optimizer(self):
        if self.opts.use_gan:
            learning_rate_d = tf.where(
                tf.greater_equal(self.global_step, self.opts.start_decay_step),
                tf.train.exponential_decay(self.opts.base_lr_d, self.global_step - self.opts.start_decay_step,
                                           self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
                self.opts.base_lr_d
            )
            learning_rate_d = tf.maximum(learning_rate_d, self.opts.lr_clip)
            add_scalar_summary('learning_rate/learning_rate_d', learning_rate_d, collection='dis')

        learning_rate_g = tf.where(
            tf.greater_equal(self.global_step, self.opts.start_decay_step),
            tf.train.exponential_decay(self.opts.base_lr_g, self.global_step - self.opts.start_decay_step,
                                       self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
            self.opts.base_lr_g
        )
        learning_rate_g = tf.maximum(learning_rate_g, self.opts.lr_clip)
        add_scalar_summary('learning_rate/learning_rate_g', learning_rate_g, collection='gen')
        gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

        with tf.control_dependencies(gen_update_ops):
            self.G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=self.opts.beta).minimize(
                self.total_gen_loss, var_list=gen_tvars, colocate_gradients_with_ops=True,
                global_step=self.global_step)
        if self.opts.use_gan:
            self.D_optimizers = tf.train.AdamOptimizer(learning_rate_d, beta1=self.opts.beta).minimize(self.D_loss,
                self.global_step, var_list=self.D.variables, name='Adam_D_X')
        # print('# of parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    def train(self):

        self.allocate_placeholders()
        self.build_model()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=None)
        self.writer = tf.summary.FileWriter(self.opts.log_dir, self.sess.graph)

        """LOAD DATA"""
        input, gt, radius = load_data(self.input, self.gt, self.opts)
        # np.savetxt('noise/gt-aug.xyz', gt[0, ...])
        # np.savetxt('noise/gt-ori.xyz', self.gt[0, ...])
        # assert 1==0

        restore_epoch = 0
        if self.opts.restore:
            restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
            self.saver.restore(self.sess, checkpoint_path)
            # self.saver.restore(self.sess, tf.train.latest_checkpoint(self.opts.log_dir))
            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
            tf.assign(self.global_step, restore_epoch).eval()
            restore_epoch += 1

        else:
            if not os.path.exists(os.path.join(self.opts.log_dir, 'plots')):
                os.makedirs(os.path.join(self.opts.log_dir, 'plots'))
            else:
                print("Current model path exists already.")
                return
            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')

        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

        step = self.sess.run(self.global_step)
        start = time()
        for epoch in range(restore_epoch, self.opts.training_epoch):
            logging.info('**** EPOCH %03d ****\t' % epoch)
            feed_dict = {self.input_x: input,
                         self.input_y: gt,
                         self.pc_radius: radius,
                         self.is_training: True}
            if self.opts.use_gan:
                # Update D network
                _, d_loss, d_summary = self.sess.run([self.D_optimizers, self.D_loss, self.d_summary_op],
                                                     feed_dict=feed_dict)
                self.writer.add_summary(d_summary, step)
            else:
                d_loss = 0

            # Update G network
            for i in range(self.opts.gen_update):
                # get previously generated images
                _, g_total_loss, summary = self.sess.run(
                    [self.G_optimizers, self.total_gen_loss, self.g_summary_op], feed_dict=feed_dict)
                self.writer.add_summary(summary, step)

            if step % self.opts.steps_per_print == 0:
                self.log_string('-----------EPOCH %d Step %d:-------------' % (epoch, step))
                self.log_string('  G_loss   : {}'.format(g_total_loss))
                self.log_string('  D_loss   : {}'.format(d_loss))
                self.log_string(' Time Cost : {}'.format(time() - start))
                start = time()
                feed_dict = {self.input_x: input,
                             self.is_training: False}

                fake_y_val = self.sess.run([self.G_y], feed_dict=feed_dict)

                fake_y_val = np.squeeze(fake_y_val)
                image_input_x = point_cloud_three_views(input[0])
                image_fake_y = point_cloud_three_views(fake_y_val[0])
                image_input_y = point_cloud_three_views(gt[0, :, 0:3])
                image_x_merged = np.concatenate([image_input_x, image_fake_y, image_input_y], axis=1)
                image_x_merged = np.expand_dims(image_x_merged, axis=0)
                image_x_merged = np.expand_dims(image_x_merged, axis=-1)
                image_x_summary = self.sess.run(self.image_x_summary,
                                                feed_dict={self.image_x_merged: image_x_merged})
                self.writer.add_summary(image_x_summary, step)

            if self.opts.visulize and (step % self.opts.steps_per_visu == 0):
                feed_dict = {self.input_x: input,
                             self.input_y: gt,
                             self.pc_radius: radius,
                             self.is_training: False}
                pcds = self.sess.run([self.visualize_ops], feed_dict=feed_dict)
                pcds = np.squeeze(pcds)  # np.asarray(pcds).reshape([3,self.opts.num_point,3])
                plot_path = os.path.join(self.opts.log_dir, 'plots',
                                         'epoch_%d_step_%d.png' % (epoch, step))
                plot_pcd_three_views(plot_path, pcds, self.visualize_titles)

            step += 1
            if (epoch % self.opts.epoch_per_save) == 0:
                self.saver.save(self.sess, os.path.join(self.opts.log_dir, 'model'), epoch)
                print(colored('Model saved at %s' % self.opts.log_dir, 'white', 'on_blue'))

    def test(self,):
        gt = self.gt[0, ...]
        print('gt', gt.shape)
        point_path = self.opts.out_folder
        self.opts.num_point = gt.shape[0]

        self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.num_point, self.opts.point_dim])
        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        Gen = Generator(self.opts, is_training, True, name='generator')
        self.pred_pc = Gen(self.inputs)

        saver = tf.train.Saver()
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
        print('checkpoint_path', checkpoint_path)
        saver.restore(self.sess, checkpoint_path)

        print(self.pred_pc.shape)
        pred = self.sess.run([self.pred_pc], feed_dict={self.inputs: np.expand_dims(gt[:, :self.opts.point_dim], 0)})
        pred = np.squeeze(np.array(pred))
        np.savetxt(point_path + '/%s.xyz' % self.opts.data_file, pred[:, :3], fmt='%.6f')

    def log_string(self, msg):
        logging.info(msg)
        self.LOG_FOUT.write(msg + "\n")
        self.LOG_FOUT.flush()
