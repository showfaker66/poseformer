# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from common.model_poseformer import PoseTransformer

# Model
print('==> Building model..')
model = PoseTransformer(num_frame=81, num_joints=17, drop_rate=0.,norm_layer=None)

dummy_input = torch.randn(512, 81, 17, 2)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
print('FLOPs = ' + str(flops/1000**3) + 'G')
# -- coding: utf-8 --
# import torch
# import torchvision
# from thop import profile
#
# # Model
# print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)
#
# dummy_input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

