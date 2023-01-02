
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy import pi
import torch.nn.utils.spectral_norm as spectral_norm
from im2scene.layers import ResnetBlockFC



class Decoder(nn.Module):
    ''' Decoder class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''

    def __init__(self, hidden_size=256, n_blocks=8, n_blocks_view=1,
                 skips=[4], use_viewdirs=True, n_freq_posenc=10,
                 n_freq_posenc_views=4,
                 z_dim=64, rgb_out_dim=128, final_sigmoid_activation=False,
                 downscale_p_by=2., positional_encoding="normal",
                 gauss_dim_pos=10, gauss_dim_view=4, gauss_std=4., cond=False,
                 **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view
        self.hidden_size = hidden_size
        self.cond = cond

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        else:
            dim_embed = 3 * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.fc_in = spectral_norm(nn.Linear(dim_embed, hidden_size))
        if z_dim > 0:
            self.fc_z = spectral_norm(nn.Linear(z_dim, hidden_size))
        '''
        self.blocks = nn.ModuleList([
            spectral_norm(nn.Linear(hidden_size, hidden_size)) for i in range(n_blocks - 1)
        ])
        '''
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size, hidden_size) for i in range(n_blocks - 1)
        ])

        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [ResnetBlockFC(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                spectral_norm(nn.Linear(dim_embed, hidden_size)) for i in range(n_skips)
            ])
        self.sigma_out = spectral_norm(nn.Linear(hidden_size, 1))

        # Feature Prediction Layers
        self.fc_z_view = spectral_norm(nn.Linear(z_dim, hidden_size))
        self.feat_view = spectral_norm(nn.Linear(hidden_size, hidden_size))
        if self.cond:
            self.cond_encoder = spectral_norm(nn.Linear(7, hidden_size))
            torch.nn.init.xavier_uniform_(self.cond_encoder.weight)
            #nn.init.normal_(self.cond_encoder.weight, 0.0, 0.005)
            #nn.init.constant_(self.cond_encoder.bias, 0.)

        self.fc_view = spectral_norm(nn.Linear(dim_embed_view, hidden_size))
        self.feat_out = spectral_norm(nn.Linear(hidden_size, rgb_out_dim))
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [spectral_norm(nn.Linear(dim_embed_view + hidden_size, hidden_size))
                 for i in range(n_blocks_view - 1)])

    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, p_in, ray_d, z_shape=None, z_app=None, cond_data=None, **kwargs):
        a = F.relu
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(p_in.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(p_in.device)
        p = self.transform_points(p_in)
        net = self.fc_in(p)
        if z_shape is not None:
            net = net * self.fc_z(z_shape).unsqueeze(1)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = layer(net)
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape).unsqueeze(1)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net).squeeze(-1)

        net = self.feat_view(net)
        if cond_data is not None and self.cond:
            net = net * self.cond_encoder(cond_data).unsqueeze(1)
        net = net + self.fc_z_view(z_app).unsqueeze(1)
        if self.use_viewdirs and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, views=True)
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return feat_out, sigma_out


class Decoder_Linear(nn.Module):
    ''' Decoder class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''

    def __init__(self, hidden_size=256, n_blocks=8, n_blocks_view=1,
                 skips=[4], use_viewdirs=True, n_freq_posenc=10,
                 n_freq_posenc_views=4,
                 z_dim=64, rgb_out_dim=128, final_sigmoid_activation=False,
                 downscale_p_by=2., positional_encoding="normal",
                 gauss_dim_pos=10, gauss_dim_view=4, gauss_std=4., cond=False,
                 **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view
        self.hidden_size = hidden_size
        self.cond = cond

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        else:
            dim_embed = 3 * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.fc_in = spectral_norm(nn.Linear(dim_embed, hidden_size))
        if z_dim > 0:
            self.fc_z = spectral_norm(nn.Linear(z_dim, hidden_size))
        
        self.blocks = nn.ModuleList([
            spectral_norm(nn.Linear(hidden_size, hidden_size)) for i in range(n_blocks - 1)
        ])


        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [spectral_norm(nn.Linear(z_dim, hidden_size)) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                spectral_norm(nn.Linear(dim_embed, hidden_size)) for i in range(n_skips)
            ])
        self.sigma_out = spectral_norm(nn.Linear(hidden_size, 1))

        # Feature Prediction Layers
        self.fc_z_view = spectral_norm(nn.Linear(z_dim, hidden_size))
        self.feat_view = spectral_norm(nn.Linear(hidden_size, hidden_size))
        if self.cond:
            self.cond_encoder = spectral_norm(nn.Linear(7, hidden_size))
            torch.nn.init.xavier_uniform_(self.cond_encoder.weight)
            #nn.init.normal_(self.cond_encoder.weight, 0.0, 0.005)
            #nn.init.constant_(self.cond_encoder.bias, 0.)

        self.fc_view = spectral_norm(nn.Linear(dim_embed_view, hidden_size))
        self.feat_out = spectral_norm(nn.Linear(hidden_size, rgb_out_dim))
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [spectral_norm(nn.Linear(dim_embed_view + hidden_size, hidden_size))
                 for i in range(n_blocks_view - 1)])

    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, p_in, ray_d, z_shape=None, z_app=None, cond_data=None, **kwargs):
        a = F.relu
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(p_in.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(p_in.device)
        p = self.transform_points(p_in)
        net = self.fc_in(p)
        if z_shape is not None:
            net = net * self.fc_z(z_shape).unsqueeze(1)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = layer(net)
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape).unsqueeze(1)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net).squeeze(-1)

        net = self.feat_view(net)
        if cond_data is not None and self.cond:
            net = net * self.cond_encoder(cond_data).unsqueeze(1)
        net = net + self.fc_z_view(z_app).unsqueeze(1)
        if self.use_viewdirs and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, views=True)
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return feat_out, sigma_out