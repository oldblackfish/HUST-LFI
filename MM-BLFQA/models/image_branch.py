import torch
import torch.nn as nn
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.relu(out)

        return out


class SemEnc(nn.Module):

    def __init__(self):
        super().__init__()

        # ---------------- Stem ----------------
        self.stem = nn.Sequential(
            nn.Conv2d(8, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )

        # ---------------- Stage 1 ----------------
        self.stage1 = ResBlock(32, 32, stride=1)

        # ---------------- Stage 2 ----------------
        self.stage2 = ResBlock(32, 64, stride=2)

        # ---------------- Stage 3 ----------------
        self.stage3 = ResBlock(64, 128, stride=2)

        # preserve distortion statistics
        self.stage4 = ResBlock(128, 128, stride=1)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x


class MergeNet(nn.Module):
    def __init__(self, an):
        super(MergeNet, self).__init__()

        self.an = an
        self.relu = nn.ReLU(inplace=True)

        self.spaconv = nn.Conv2d(8, 8, 3, 1, 1)
        self.spabn = nn.BatchNorm2d(8)

        self.angconv = nn.Conv2d(8, 8, 3, 1, 1)
        self.angbn = nn.BatchNorm2d(8)

    def forward(self, x):
        # ---------- spatial conv ----------
        out = self.relu(self.spabn(self.spaconv(x)))

        N_total, c, h, w = out.shape
        N = N_total // (self.an * self.an)

        # ---------- SAI stack → angular image ----------
        out = rearrange(
            out, '(n u v) c h w -> (n h w) c u v', n=N, u=self.an, v=self.an
        )

        # ---------- angular conv ----------
        out = self.relu(self.angbn(self.angconv(out)))

        # ---------- angular image → SAI stack ----------
        out = rearrange(
            out, '(n h w) c u v -> (n u v) c h w', n=N, h=h, w=w
        )

        return out


# ============================================================
# Alternating Stage  ⭐ NEW (核心重构)
# ============================================================

class AltBlockStage(nn.Module):
    """
    Input  : MacPI  [N, C, H*an, W*an]
    Output : MacPI  [N, C, H*an, W*an]
    """

    def __init__(self, angRes, layer_num):
        super().__init__()

        self.an = angRes
        self.relu = nn.LeakyReLU()

        # shallow feature extraction
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(8)

        # alternating blocks
        layers = []
        for _ in range(layer_num):
            layers.append(MergeNet(angRes))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        # ==================================================
        # MacPI → SAI stack
        # ==================================================
        x = rearrange(
            x, 'n c (h u) (w v) -> (n u v) c h w', u=self.an, v=self.an
        )

        x = self.relu(self.bn0(self.conv0(x)))

        # ==================================================
        # alternating spatial-angular learning
        # ==================================================
        x = self.blocks(x)

        # ==================================================
        # SAI stack → MacPI
        # ==================================================
        N_total, C, h, w = x.shape
        N = N_total // (self.an * self.an)

        x = rearrange(
            x, '(n u v) c h w -> n c (h u) (w v)', n=N, u=self.an, v=self.an
        )

        return x


class ImageBranch(nn.Module):

    def __init__(self, angRes, layer):
        super(ImageBranch, self).__init__()

        # Angular-Spatial Interaction (ASI) block
        self.altstage = AltBlockStage(angRes, layer)

        # Semantic Encoder
        self.senet = SemEnc()

    def forward(self, x):
        # ---------- Angular-Spatial Interaction (ASI) block ----------
        x = self.altstage(x)

        # ---------- semantic encoder ----------
        x = self.senet(x)

        return x


if __name__ == "__main__":
    from thop import profile

    net = ImageBranch(angRes=5, layer=1).cuda()
    input1 = torch.randn(1, 1, 5 * 32, 5 * 32).cuda()
    flops, params = profile(net, inputs=(input1,))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
