
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(channels + 2*growth_channels, growth_channels, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(channels + 3*growth_channels, growth_channels, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(channels + 4*growth_channels, channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RestorationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64, num_rrdb=6, upscale=2):
        super().__init__()
        self.upscale = upscale
        
        self.conv_first = nn.Conv2d(in_channels, features, 3, 1, 1, bias=True)
        self.body = nn.ModuleList([RRDB(features) for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        
        self.upconv1 = nn.Conv2d(features, features * (upscale ** 2), 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.upconv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(features, out_channels, 3, 1, 1, bias=True)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # Bicubic upsample for residual connection
        x_up = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        
        feat = self.conv_first(x)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        out = self.upconv1(feat)
        out = self.pixel_shuffle(out)
        out = self.upconv2(out)
        out = self.conv_last(out)
        
        return out + x_up