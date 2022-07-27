from __future__ import division
import time
import torch
import numpy as np
import cv2
import os
import sys
import random
import pickle as pkl
import argparse

from lib.yolov3.util import *
from demo.lib.yolov3.darknet import Darknet
from demo.lib.yolov3 import preprocess

cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.join(cur_dir, '../../../')
chk_root = os.path.join(project_root, 'checkpoint/')
data_root = os.path.join(project_root, 'data/')


sys.path.insert(0, project_root)
sys.path.pop(0)


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    ori_img = img
    dim = ori_img.shape[1], ori_img.shape[0]
    img = cv2.resize(ori_img, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, ori_img, dim


def write(x, img, colors):
    x = [int(i) for i in x]
    c1 = tuple(x[0:2])
    c2 = tuple(x[2:4])

    label = 'People {}'.format(0)
    color = (0, 0, 255)
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def arg_parse():
    """"
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument('--confidence', dest='confidence', type=float, default=0.70,
                        help='Object Confidence to filter predictions')
    parser.add_argument('--nms-thresh', dest='nms_thresh', type=float, default=0.4, help='NMS Threshold')
    parser.add_argument('--reso', dest='reso', default=416, type=int, help='Input resolution of the network. '
                        'Increase to increase accuracy. Decrease to increase speed. (160, 416)')
    parser.add_argument('-wf', '--weight-file', type=str, default=r'I:\Human_pose_estimation\MHFormer-main\demo\lib\checkpoint\\yolov3.weights', help='The path'
                        'of model weight file')
    parser.add_argument('-cf', '--cfg-file', type=str, default=cur_dir + '/cfg/yolov3.cfg', help='weight file')
    parser.add_argument('-a', '--animation', action='store_true', help='output animation')
    parser.add_argument('-v', '--video', type=str, default='camera', help='The input video path')
    parser.add_argument('-i', '--image', type=str, default=cur_dir + '/data/dog-cycle-car.png',
                        help='The input video path')
    parser.add_argument('-np', '--num-person', type=int, default=1, help='number of estimated human poses. [1, 2]')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    
    return parser.parse_args()


def load_model(args=None, CUDA=None, inp_dim=416):
    if args is None:
        args = arg_parse()

    if CUDA is None:
        CUDA = torch.cuda.is_available()

    # Set up the neural network
    model = Darknet(args.cfg_file)
    model.load_weights(args.weight_file)
    # print("YOLOv3 network successfully loaded")

    model.net_info["height"] = inp_dim
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    return model

# img: 720, 1280, 3   model: 加载了权重的yolov3人体检测网络    置信度: 0.3
def yolo_human_det(img, model=None, reso=416, confidence=0.30):
    args = arg_parse()
    # args.reso = reso
    inp_dim = reso
    num_classes = 80

    CUDA = torch.cuda.is_available()
    if model is None:  # 否
        model = load_model(args, CUDA, inp_dim)

    if type(img) == str:  # 否
        assert os.path.isfile(img), 'The image path does not exist'
        img = cv2.imread(img)

    img, ori_img, img_dim = preprocess.prep_image(img, inp_dim)    # 1, 3, 416, 416  720, 1280, 3   1280, 720
    img_dim = torch.FloatTensor(img_dim).repeat(1, 2)  # img_dim.shape: 1, 2 -> 1, 4    data: 1280, 720  -> 1280, 720, 1280, 720

    with torch.no_grad():
        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()
        output = model(img, CUDA)   # 1, 10647, 85
        # args.nms_thresh = 0.4 经过IOU排序处理后留下的预测框，再删除置信度小于阈值的预测框
        # 0.0000, 189.9222, 125.0656, 247.6008, 316.5523,  0.9999,  1.0000,  0.0000
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=args.nms_thresh, det_hm=True)  # 1, 8

        if len(output) == 0:
            return None, None

        img_dim = img_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / img_dim, 1)[0].view(-1, 1)  # 0.325

        output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim[:, 0].view(-1, 1)) / 2  # 1, 2  189.9222和247.6008
        output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim[:, 1].view(-1, 1)) / 2  # 1, 2  34.0656和225.5523
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim[i, 1])

    bboxs = []
    scores = []
    for i in range(len(output)):
        item = output[i]       # 0.0000, 584.3760, 104.8172, 761.8487, 694.0070,   0.9999,   1.0000,  0
        bbox = item[1:5].cpu().numpy()
        # conver float32 to .2f data
        bbox = [round(i, 2) for i in list(bbox)]
        score = item[5].cpu().numpy()
        bboxs.append(bbox)
        scores.append(score)
    scores = np.expand_dims(np.array(scores), 1)
    bboxs = np.array(bboxs)

    return bboxs, scores
# bboxs: 584.3760, 104.8172, 761.8487, 694.0070   scores: 0.9999
