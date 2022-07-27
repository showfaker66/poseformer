# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import os.path as osp
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.realpath(__file__)), '..'))
from utils.transforms import transform_preds
sys.path.pop(0)


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)  # 1, 17  沿轴向得到最大值的索引
    maxvals = np.amax(heatmaps_reshaped, 2)  # 1, 17 沿轴向得到最大值

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)  # np.title(idx,(1, 1, 2))沿第3个维度扩大2倍 （1, 17, 2）

    preds[:, :, 0] = (preds[:, :, 0]) % width  # [:, :, 0] 取三维矩阵中第一维的所有数据 索引%72取余
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)   # 索引/72 取整    所以现在 preds为(1, 17, 2)  17行2列  第一列为取余, 第二列为取整数

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2)) # np.greater 判断参数1是否大于参数2
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals  # 从热图中得到最大预测坐标


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)  # 坐标, 热图关键点分数  (1, 17, 2)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing 后处理
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back 通过热图预测的最大概率的坐标，原预测框的中心坐标，缩放比例，热图宽高 -> 回归出在原frame中的关键点坐标
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
