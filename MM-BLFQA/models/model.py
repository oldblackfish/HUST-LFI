import torch
import torch.nn as nn

from .image_branch import ImageBranch
from .video_branch import VideoBranch
from .moe import MoE


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.img_branch = ImageBranch(angRes=5, layer=1)
        self.vid_branch = VideoBranch(angRes=5)

        self.gap3D = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.gap2D = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        # regression heads
        self.task_main_i = MoE(channels=128, out_dim=1)
        self.task_main_v = MoE(channels=128, out_dim=1)
        self.task_aux1 = nn.Linear(128*2, 10)
        self.task_aux2 = nn.Linear(128*2, 5)
        self.fc = nn.Linear(128*2, 2)

    def forward(self, x):
        f_i = self.img_branch(x)
        f_v = self.vid_branch(x)

        f_i = self.flat(self.gap2D(f_i))
        f_v = self.flat(self.gap3D(f_v))

        f_out = torch.cat((f_i, f_v), dim=1)

        q_i = self.task_main_i(f_i)
        q_v = self.task_main_v(f_v)

        weights = self.fc(f_out).softmax(dim=1)
        q = torch.sum(weights * torch.cat([q_i, q_v], dim=1), dim=1, keepdim=True)
        c = self.task_aux1(f_out)
        l = self.task_aux2(f_out)

        return q, c, l


if __name__ == "__main__":
    from thop import profile

    net = Network().cuda()

    input1 = torch.randn(1, 1, 5 * 32, 5 * 32).cuda()
    flops, params = profile(net, inputs=(input1,))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
