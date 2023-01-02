import torch.nn as nn
import torch
from math import log2
from im2scene.layers import Blur
import torch.nn.utils.spectral_norm as spectral_norm
from im2scene.layers import ResnetBlock


class NeuralRenderer(nn.Module):
    ''' Neural renderer class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=256, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=128, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False, cond=False, #False였다
            **kwargs):
        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        n_blocks = int(log2(img_size) - 4)

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = spectral_norm(nn.Conv2d(input_dim, n_feat, 1, 1, 0))

        '''
        self.conv_layers = nn.ModuleList(
            [spectral_norm(nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1))] +
            [spectral_norm(nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1))
                for i in range(0, n_blocks - 1)]
        )
        '''
        self.conv_layers = nn.ModuleList(
            [ResnetBlock(n_feat, n_feat // 2)] +
            [ResnetBlock(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat))
                for i in range(0, n_blocks - 1)]
        )
        '''
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [spectral_norm(nn.Conv2d(input_dim, out_dim, 3, 1, 1))] +
                [spectral_norm(nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1)) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = spectral_norm(nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1))
        '''
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [ResnetBlock(input_dim, out_dim)] +
                [ResnetBlock(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = spectral_norm(nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1))
        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])

        if cond:
            self.cond_encoder = spectral_norm(nn.Linear(40, n_feat))
        self.actvn = nn.ReLU()

    def forward(self, x, cond_data=None):
        net = self.conv_in(x)
        if cond_data is not None:
            net = net * self.cond_encoder(cond_data).unsqueeze(2).unsqueeze(2)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_2(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            #net = self.actvn(hid)
            net = hid

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb


class NeuralRenderer_Linear(nn.Module):
    ''' Neural renderer class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=256, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=128, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False, cond=False, #False였다
            **kwargs):
        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        n_blocks = int(log2(img_size) - 4)

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = spectral_norm(nn.Conv2d(input_dim, n_feat, 1, 1, 0))
        
        self.conv_layers = nn.ModuleList(
            [spectral_norm(nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1))] +
            [spectral_norm(nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1))
                for i in range(0, n_blocks - 1)]
        )

        
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [spectral_norm(nn.Conv2d(input_dim, out_dim, 3, 1, 1))] +
                [spectral_norm(nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1)) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = spectral_norm(nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1))

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])

        if cond:
            self.cond_encoder = spectral_norm(nn.Linear(40, n_feat))
        self.actvn = nn.ReLU()

    def forward(self, x, cond_data=None):

        net = self.conv_in(x)
        if cond_data is not None:
            net = net * self.cond_encoder(cond_data).unsqueeze(2).unsqueeze(2)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_2(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            #net = self.actvn(hid)
            net = hid

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb
