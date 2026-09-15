"""D-LinkNet34 for fixed-width thin-ribbon segmentation at 512x512.

Every decoder stage doubles resolution and the final 2x is a learned sub-pixel
convolution, so the 512x512 logits are produced by the network rather than by a
bilinear stretch of a 128x128 map.

Skips are additive (LinkNet), not concatenated: for a constant-width target the
skip restores coordinates, not semantics, and addition costs half the activation
memory -- which is what buys batch 16 with a full-resolution decoder in 8.5 GB.

Convolutions run before upsampling so the interpolated tensor carries the reduced
channel count, not the input's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


def conv_bn(cin, cout, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


class DLinkNet34(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.encoder = get_encoder("resnet34", in_channels=3, depth=5, weights=None)
        if weights is not None:
            from safetensors.torch import load_file
            sd = load_file(weights)
            self.encoder.load_state_dict(sd, strict=False)   # returns None in smp; do not unpack
            have = set(self.encoder.state_dict()) - set(sd)
            assert all(k.endswith("num_batches_tracked") for k in have), sorted(have)[:5]

        c = self.encoder.out_channels          # (3, 64, 64, 128, 256, 512)

        # Cascaded dilated context at OS32 = 16x16. Rate 8 would span 17 px on a
        # 16 px map -- every outer tap falls in zero padding -- so the cascade
        # stops at 4. ERF added: 2*(1+2+4)+1 = 15 at OS32 = 480 px at input.
        self.d1 = conv_bn(c[5], c[5], 1)
        self.d2 = conv_bn(c[5], c[5], 2)
        self.d3 = conv_bn(c[5], c[5], 4)

        self.up4 = conv_bn(c[5], c[4])
        self.up3 = conv_bn(c[4], c[3])
        self.up2 = conv_bn(c[3], c[2])
        self.up1 = conv_bn(c[2], c[1])
        self.head = conv_bn(c[1], 32)
        # 4 channels at 256 -> PixelShuffle(2) -> 1 channel at 512. A learned 2x
        # with no full-resolution 32-channel intermediate and, unlike a stride-2
        # transposed conv with k=3, no period-2 overlap modulation -- which on a
        # 4 px ribbon would be a ~30% width ripple.
        self.out = nn.Conv2d(32, 4, 3, padding=1)

    def forward(self, x):
        _, f1, f2, f3, f4, f5 = self.encoder(x)
        a = self.d1(f5)
        b = self.d2(a)
        c = self.d3(b)
        x = f5 + a + b + c
        x = F.interpolate(self.up4(x), scale_factor=2, mode="nearest") + f4
        x = F.interpolate(self.up3(x), scale_factor=2, mode="nearest") + f3
        x = F.interpolate(self.up2(x), scale_factor=2, mode="nearest") + f2
        x = F.interpolate(self.up1(x), scale_factor=2, mode="nearest") + f1
        return F.pixel_shuffle(self.out(self.head(x)), 2)