import torch
import torch.nn as nn
from common.trans import Transformer as Transformer_s
from common.trans_hypothesis import Transformer



# model_pos_train = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=32, depth=4,
#         num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.2)

class Model(nn.Module):
    def __init__(self, layers=4, channel=256, d_hid=192, length=81, num_joints_in=17, num_joints_out = 17, out_joints=17):
        super().__init__()

        self.out_joints = out_joints
        self.num_joints_out = num_joints_out
        self.num_joints_in = num_joints_in

        self.norm_1 = nn.LayerNorm(length)
        self.norm_2 = nn.LayerNorm(length)
        self.norm_3 = nn.LayerNorm(length)

        self.trans_auto_1 = Transformer_s()
        self.trans_auto_2 = Transformer_s()
        self.trans_auto_3 = Transformer_s()
        # 4, length, length * 2, length = 2 * self.num_joints_in, h = 9

        self.encoder_1 = nn.Sequential(nn.Conv1d(2 * self.num_joints_in, channel, kernel_size=1))
        self.encoder_2 = nn.Sequential(nn.Conv1d(2 * self.num_joints_in, channel, kernel_size=1))
        self.encoder_3 = nn.Sequential(nn.Conv1d(2 * self.num_joints_in, channel, kernel_size=1))

        self.Transformer = Transformer(layers, channel * 3, d_hid, length=length)

        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel * 3, momentum=0.1),
            nn.Conv1d(channel * 3, 3 * self.num_joints_out, kernel_size=1)
        )

    def forward(self, x):
        # x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()  # 输入 128, 2, 351, 17, 1  -> 128, 351, 17, 2
        x = x.view(x.shape[0], x.shape[1], -1)  # 16, 81, 34
        x = x.permute(0, 2, 1).contiguous()  # 16, 34, 81

        ## MHG
        x_1 = x + self.trans_auto_1(self.norm_1(x))  # 16, 34, 81
        x_2 = x_1 + self.trans_auto_2(self.norm_2(x_1))  # 16, 34, 81
        x_3 = x_2 + self.trans_auto_3(self.norm_3(x_2))  # 16, 34, 81

        ## Embedding
        x_1 = self.encoder_1(x_1)  # 16, 512, 81
        x_1 = x_1.permute(0, 2, 1).contiguous()  # 16, 81, 512

        x_2 = self.encoder_2(x_2)  # 16, 512, 81
        x_2 = x_2.permute(0, 2, 1).contiguous()  # 16, 81, 512

        x_3 = self.encoder_3(x_3)  # 16, 512, 81
        x_3 = x_3.permute(0, 2, 1).contiguous()  # 16, 81, 512

        ## SHR & CHI
        x = self.Transformer(x_1, x_2, x_3)  # x1 x2 x3      16, 81, 256   f:351   ->  16, 81, 768

        ## Head
        # x = x.permute(0, 2, 1).contiguous()  # 8, 1536, 351
        # x = self.fcn(x)  # 8, 51, 351
        #
        # x = x.view(x.shape[0], self.num_joints_out, -1, x.shape[2])  # 8, 17, 3, 351
        # x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1)

        return x






