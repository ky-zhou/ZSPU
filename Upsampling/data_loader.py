# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:11 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : data_loader.py

import numpy as np
import h5py
import queue
import threading
from Common import point_operation


def normalize_point_cloud(input):
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance


def batch_sampling(input_data, num):
    B, N, C = input_data.shape
    out_data = np.zeros([B, num, C])
    for i in range(B):
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx = idx[:num]
        out_data[i, ...] = input_data[i, idx]
    return out_data


def load_data(input_s, gt_s, opts):
    assert len(input_s) == len(gt_s)
    input, gt = input_s.copy(), gt_s.copy()
    radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    print("DATA input and gt:", input.shape, gt.shape)
    input = point_operation.jitter_perturbation_point_cloud(input, sigma=opts.jitter_sigma, clip=opts.jitter_max)
    input, gt = point_operation.rotate_point_cloud_and_gt(input, gt)
    input, gt, scales = point_operation.random_scale_point_cloud_and_gt(input, gt, scale_low=0.8, scale_high=1.2)
    radius = radius * scales

    return input, gt, radius






