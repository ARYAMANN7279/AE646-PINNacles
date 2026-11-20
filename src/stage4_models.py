"""
Stage 4 baselines for the kappa -> u operator, same input/output convention as models.py:
input (B, H, W, 3) = (kappa, x, y) channels-last, output (B, H, W, 1).

UNet2d      convolutional encoder-decoder with skip connections (4 levels, GroupNorm, GELU). Works on any
            grid whose side is divisible by 2**levels, so it can also be evaluated zero-shot on other grids.
DeepONet2d  branch net = CNN encoder of the permeability map (adaptive pooling, so any input grid works),
            trunk net = MLP of the query coordinates (x, y); u(x, y) = sum_k b_k(kappa) t_k(x, y) + bias.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, groups=8):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.GELU())


class UNet2d(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, base=32, levels=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(levels + 1)]          # 32, 64, 128, 256, 512
        self.levels = levels
        self.enc = nn.ModuleList([_block(input_channels if i == 0 else chs[i - 1], chs[i]) for i in range(levels)])
        self.mid = _block(chs[levels - 1], chs[levels])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(levels))])
        self.dec = nn.ModuleList([_block(2 * chs[i], chs[i]) for i in reversed(range(levels))])
        self.out = nn.Conv2d(chs[0], output_channels, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec in zip(self.up, self.dec):
            x = up(x); x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.out(x).permute(0, 2, 3, 1)


class DeepONet2d(nn.Module):
    def __init__(self, p=128, base=32, trunk_width=128):
        super().__init__()
        c = [1, base, 2 * base, 4 * base, 8 * base]
        layers = []
        for i in range(4):                                         # 64 -> 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1), nn.GELU()]
        self.branch_conv = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(4))
        self.branch_fc = nn.Sequential(nn.Flatten(), nn.Linear(c[4] * 16, 512), nn.GELU(), nn.Linear(512, p))
        self.trunk = nn.Sequential(nn.Linear(2, trunk_width), nn.GELU(), nn.Linear(trunk_width, trunk_width), nn.GELU(),
                                   nn.Linear(trunk_width, trunk_width), nn.GELU(), nn.Linear(trunk_width, p))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = self.branch_fc(self.branch_conv(x[..., :1].permute(0, 3, 1, 2)))          # (B, p)
        t = self.trunk(x[..., 1:3])                                                   # (B, H, W, p)
        return (torch.einsum("bp,bhwp->bhw", b, t) + self.bias).unsqueeze(-1)


def get_stage4_model(arch, **kw):
    if arch == "unet":
        return UNet2d(**kw)
    if arch == "deeponet":
        return DeepONet2d(**kw)
    raise ValueError(arch)
