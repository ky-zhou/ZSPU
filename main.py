import tensorflow as tf
from Upsampling.model import Model
from Upsampling.configs import FLAGS
from Common.point_operation import nonuniform_sampling, guass_noise_point_cloud
from datetime import datetime
import os
import logging
import pprint
from time import time
import numpy as np
import h5py
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
pp = pprint.PrettyPrinter()


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def run():
    FLAGS.point_dim = 6 if FLAGS.use_norm else 3
    # train_file, train_index = FLAGS.data_file.split('_')[0], FLAGS.data_file.split('_')[1]
    train_file = FLAGS.data_file

    start_time = time()
    if FLAGS.use_data == 0:
        FLAGS.data_dir = 'data/pu/'
        FLAGS.train_file = os.path.join(FLAGS.data_dir, '%s.xyz' % train_file)
        gt = np.loadtxt(FLAGS.train_file, delimiter=' ')[:4096, :FLAGS.point_dim]
    elif FLAGS.use_data == 1:
        FLAGS.data_dir = 'data/ps/'
        FLAGS.train_file = os.path.join(FLAGS.data_dir, '%s.xyz' % train_file)
        gt = np.loadtxt(FLAGS.train_file, delimiter=' ')[:4096, :FLAGS.point_dim]
    elif FLAGS.use_data == 2:
        FLAGS.data_dir = 'data/kitti/'
        FLAGS.train_file = os.path.join(FLAGS.data_dir, '%s.xyz' % train_file)
        print('Opening:', FLAGS.train_file)
        gt = np.loadtxt(FLAGS.train_file)[:, :FLAGS.point_dim]
    elif FLAGS.use_data == 3:
        FLAGS.data_dir = 'data/mpu/'
        FLAGS.train_file = os.path.join(FLAGS.data_dir, '%s.xyz' % train_file)
        gt = np.loadtxt(FLAGS.train_file, delimiter=' ')[:, :FLAGS.point_dim]
    print('train_file:', FLAGS.train_file)
    gt = np.expand_dims(gt[:, :], 0)
    n_pcd = (gt.shape[1]//FLAGS.up_ratio) * FLAGS.up_ratio
    b, down_size = FLAGS.batch_size, n_pcd//FLAGS.up_ratio
    gt = gt[:, :n_pcd, :]
    gt = np.tile(gt, (b, 1, 1))
    if FLAGS.use_noise:
        sigma = FLAGS.noise
        gt = guass_noise_point_cloud(gt, sigma=sigma)
        np.savetxt('noise/%s_in-%.3f-%s.xyz' % (train_file, sigma, FLAGS.down_kernel), gt[0, ...])
    if FLAGS.down_kernel == 'fps':
        print("=============Downsample: ideal kernel (FPS)============")
        gt_pl = tf.placeholder(tf.float32, shape=[b, n_pcd, FLAGS.point_dim])
        fps_pl = gather_point(gt_pl, farthest_point_sample(down_size, gt_pl))
        with tf.Session() as sess:
            input = sess.run([fps_pl], feed_dict={gt_pl: gt})
            input = np.squeeze(input, axis=0)
            sess.close()
    else:
        print("=============Downsample: non-ideal kernel (Random)============")
        input = np.zeros((b, down_size, FLAGS.point_dim))
        for i in range(b):
            idx = nonuniform_sampling(n_pcd, sample_num=down_size)
            input[i, ...] = gt[i, idx, :]

    if not FLAGS.restore:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        FLAGS.log_dir = os.path.join(FLAGS.log_dir, FLAGS.data_file)
        try:
            os.makedirs(FLAGS.log_dir)
        except os.error:
            pass
    FLAGS.in_folder = os.path.join(FLAGS.data_dir, 'input')
    if not os.path.exists(FLAGS.in_folder):
        os.makedirs(FLAGS.in_folder)
    np.savetxt(os.path.join(FLAGS.in_folder, '%s.xyz' % FLAGS.data_file), input[0, ...])
    print("Input has size:", input.shape)
    print("GT has size:", gt.shape)
    FLAGS.out_folder = os.path.join(FLAGS.data_dir, 'output')
    if not os.path.exists(FLAGS.out_folder):
        os.makedirs(FLAGS.out_folder)
    if not os.path.exists(os.path.join(FLAGS.log_dir, 'code/')):
        os.makedirs(os.path.join(FLAGS.log_dir, 'code/'))
        os.system('cp -r Common/* %s' % (os.path.join(FLAGS.log_dir, 'code/')))  #
        os.system('cp -r Upsampling/* %s' % (os.path.join(FLAGS.log_dir, 'code/')))  #

    pp.pprint(FLAGS)
    # open session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        model = Model(FLAGS, sess, n_pcd, input, gt)
        model.train()
        model.test()
        sess.close()
    print('All time cost:', time() - start_time)


def main(unused_argv):
    run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
