from __future__ import annotations

import copy
import math
from logging import warning

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except OSError:
    pass
from typing import TYPE_CHECKING

from nect.network.kplanes import init_grid_param, interpolate_ms_features

if TYPE_CHECKING:
    from nect.config import (
        HashEncoderConfig,
        KPlanesEncoderConfig,
        MLPNetConfig,
        PirateNetConfig,
    )


class HashGrid(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
    ):
        super().__init__()
        self.include_identity = network_config.include_identity
        if self.include_identity:
            encoding = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    },
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Identity"
                    }
                ]
            }
        else:
            encoding = encoding_config.get_encoder_config()
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=3 + (3 if self.include_identity else 0),
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )   

    def forward(self, x):
        if self.include_identity:
            inputs = torch.cat([x, x], dim=-1)
        else:
            inputs = x
        out = self.net(inputs)
        return out

class MultiResolutionHashTimeDoubleNetwork(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: MLPNetConfig) -> None:
        super(MultiResolutionHashTimeDoubleNetwork, self).__init__()
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )
        self.time_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=1,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )

    def forward(self, x, t):
        x_out = self.model(x)
        t_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        time_out = self.time_model(t_input)
        return x_out + time_out


class MultiResolutionHashTimeDoubleEncoderNetwork(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: MLPNetConfig) -> None:
        super(MultiResolutionHashTimeDoubleEncoderNetwork, self).__init__()
        self.include_adaptive_skip = network_config.include_adaptive_skip
        if network_config.include_adaptive_skip:
            self.encoder_static = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.encoder_dynamic = tcnn.Encoding(
                n_input_dims=4,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.skip_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            original_n_hidden_layers = network_config.n_hidden_layers
            network_config.n_hidden_layers = math.floor(original_n_hidden_layers / 2)
            n_neurons = self.encoder_dynamic.n_output_dims + self.encoder_static.n_output_dims
            network_config.n_neurons = n_neurons
            if n_neurons not in [16, 32, 64, 128]:
                network_config.otype = "CutlassMLP"
            original_output_activation = network_config.output_activation
            network_config.output_activation = network_config.activation
            self.net_1 = tcnn.Network(
                n_input_dims=n_neurons,
                n_output_dims=n_neurons,
                network_config=network_config.get_network_config(),
            )
            network_config_2 = copy.deepcopy(network_config)
            network_config_2.n_hidden_layers = original_n_hidden_layers - network_config.n_hidden_layers
            network_config_2.output_activation = original_output_activation
            self.net_2 = tcnn.Network(
                n_input_dims=n_neurons,
                n_output_dims=1,
                network_config=network_config_2.get_network_config(),
                seed=1,
            )

        else:
            encoding = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    },
                    {
                        "n_dims_to_encode": 4,
                        **encoding_config.get_encoder_config()
                    },
                ]
            }
            self.net = tcnn.NetworkWithInputEncoding(
                n_input_dims=7,
                n_output_dims=1,
                network_config=network_config.get_network_config(),
                encoding_config=encoding
            )

    def forward(self, x, t):
        t_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        if self.include_adaptive_skip:
            static_features = self.encoder_static(x)
            dynamic_features = self.encoder_dynamic(t_input)
            feature_tensor = torch.cat([static_features, dynamic_features], dim=-1)
            out = self.net_1(feature_tensor)
            return self.net_2(out * self.skip_alpha + (1 - self.skip_alpha) * feature_tensor)
        return self.net(torch.cat([x, t_input], dim=-1))


class MultiResolutionHashTimeDoubleEncoderDoubleNetwork(nn.Module):
    def __init__(self, encoding_config, network_config) -> None:
        super(MultiResolutionHashTimeDoubleEncoderDoubleNetwork, self).__init__()
        print(encoding_config.get_encoder_config())
        print(network_config.get_network_config())
        self.static = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=64,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )
        self.encoder_dynamic = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=64,
            encoding_config=encoding_config.get_encoder_config(),
            network_config=network_config.get_network_config(),
        )
        self.include_adaptive_skip = network_config.include_adaptive_skip
        if network_config.include_adaptive_skip:
            self.skip_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            original_n_hidden_layers = network_config["n_hidden_layers"]
            if original_n_hidden_layers < 2:
                original_n_hidden_layers = 2
                warning("Original number of hidden layers is less than 2, setting to 2")
            network_config["n_hidden_layers"] = max(math.floor(original_n_hidden_layers / 2), 1)
            original_output_activation = network_config["output_activation"]
            network_config["output_activation"] = original_output_activation
            self.net_1 = tcnn.Network(
                n_input_dims=128,
                n_output_dims=128,
                network_config=network_config.get_network_config(),
            )
            network_config_2 = copy.deepcopy(network_config)
            network_config_2["n_hidden_layers"] = original_n_hidden_layers - network_config["n_hidden_layers"]
            self.net_2 = tcnn.Network(
                n_input_dims=128,
                n_output_dims=1,
                network_config=network_config_2,
                seed=1,
            )
        else:
            self.net = tcnn.Network(
                n_input_dims=128,
                n_output_dims=1,
                network_config=network_config.get_network_config(),
            )

    def forward(self, x, t):
        static_features = self.static(x)
        t_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        dynamic_features = self.encoder_dynamic(t_input)
        features = torch.cat([static_features, dynamic_features], dim=-1)
        if self.include_adaptive_skip:
            out = self.net_1(features)
            out = self.net_2(out * self.skip_alpha + (1 - self.skip_alpha) * features)
        else:
            out = self.net(features)
        return out


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GatedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, input1, input2):
        out = F.leaky_relu(self.linear(x))
        return out * input1 + (1 - out) * input2


class AdaptiveBlock(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, alfa_init) -> None:
        super(AdaptiveBlock, self).__init__()
        self.dense1 = GatedLinear(n_input_dims, n_input_dims)
        self.dense2 = GatedLinear(n_input_dims, n_input_dims)
        self.dense3 = nn.Linear(n_input_dims, n_input_dims)
        self.alfa = nn.Parameter(torch.tensor(alfa_init), requires_grad=True)

    def forward(self, x, inpu1, input2):
        dense1 = self.dense1(x, inpu1, input2)
        dense2 = self.dense2(dense1, inpu1, input2)
        dense3 = F.leaky_relu(self.dense3(dense2))
        dense3 = dense3 * self.alfa + (1 - self.alfa) * x
        return dense3


class PirateNetwork(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, n_modules, alfa_init) -> None:
        super(PirateNetwork, self).__init__()
        self.dense1 = nn.Linear(n_input_dims, n_input_dims)
        self.dense2 = nn.Linear(n_input_dims, n_input_dims)
        self.adaptive_block = nn.ModuleList(
            AdaptiveBlock(n_input_dims, n_input_dims, alfa_init) for _ in range(n_modules)
        )
        self.out = nn.Linear(n_input_dims, n_output_dims, bias=True)

    def forward(self, x):
        dense1 = F.leaky_relu(self.dense1(x))
        dense2 = F.leaky_relu(self.dense2(x))
        for block in self.adaptive_block:
            x = block(x, dense1, dense2)
        out = self.out(x)
        if out.max() < 0:
            return out
        return F.relu(out)


class PirateNet(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: PirateNetConfig) -> None:
        super(PirateNet, self).__init__()
        self.hash_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.net = PirateNetwork(
            n_input_dims=self.hash_encoder.n_output_dims,
            n_output_dims=1,
            n_modules=network_config.n_modules,
            alfa_init=network_config.alfa_init,
        )

    def forward(self, x):
        x_encoded = self.hash_encoder(x)
        out = self.net(x_encoded)
        return out


class DynamicPirateNet(nn.Module):
    def __init__(self, encoding_config: HashEncoderConfig, network_config: PirateNetConfig) -> None:
        super(DynamicPirateNet, self).__init__()
        self.hash_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.dynamic_encoder = tcnn.Encoding(
            n_input_dims=4,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.net = PirateNetwork(
            n_input_dims=self.hash_encoder.n_output_dims + self.dynamic_encoder.n_output_dims,
            n_output_dims=1,
            n_modules=network_config.n_modules,
            alfa_init=network_config.alfa_init,
        )

    def forward(self, x, t):
        x_encoded = self.hash_encoder(x)
        t_encoded = self.dynamic_encoder(torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1))
        out = self.net(torch.cat([x_encoded, t_encoded], dim=-1))
        return out

class QuadCubesOld(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
        prior=False,
        concat=True,
    ):
        super().__init__()
        self.concat = concat
        self.scale_factor = 100
        self.static = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config.get_encoder_config(),
        )
        self.prior = prior
        if not prior:
            self.xyt = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.xzt = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
            self.yzt = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config.get_encoder_config(),
            )
        self.include_identity = network_config.include_identity
        additional_parameters = 4 if network_config.include_identity else 0
        self.net = tcnn.Network(
            n_input_dims=self.static.n_output_dims * (4 if self.concat else 1) + additional_parameters,
            n_output_dims=1,
            network_config=network_config.get_network_config(),
        )
        if self.concat is False:
            for encoder in [self.static, self.xyt, self.xzt, self.yzt]:
                for param in encoder.parameters():
                    torch.nn.init.ones_(param.data)

    def forward(self, x, t):
        if self.include_identity and self.concat:
            xyzt_input = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)
        static_encoded = self.static(x)
        if not self.prior:
            xyt_encoded = self.xyt(
                torch.cat(
                    [x[..., [1, 2]], torch.full((x.size(0), 1), t, device=x.device)],
                    dim=1,
                )
            )
            xzt_encoded = self.xzt(
                torch.cat(
                    [x[..., [0, 2]], torch.full((x.size(0), 1), t, device=x.device)],
                    dim=1,
                )
            )
            yzt_encoded = self.yzt(
                torch.cat(
                    [x[..., [0, 1]], torch.full((x.size(0), 1), t, device=x.device)],
                    dim=1,
                )
            )
        else:
            xyt_encoded = torch.zeros_like(static_encoded)
            xzt_encoded = torch.zeros_like(static_encoded)
            yzt_encoded = torch.zeros_like(static_encoded)
        if self.concat:
            if self.include_identity:
                to_mlp = torch.cat([static_encoded, xyt_encoded, xzt_encoded, yzt_encoded, xyzt_input], dim=-1)
            else:
                to_mlp = torch.cat([static_encoded, xyt_encoded, xzt_encoded, yzt_encoded], dim=-1)
        else:
            if self.include_identity:
                to_mlp = torch.cat([static_encoded * xyt_encoded * xzt_encoded * yzt_encoded, xyzt_input], dim=-1)
            else:
                to_mlp = static_encoded * xyt_encoded * xzt_encoded * yzt_encoded
        out = self.net(to_mlp)
        return out


class QuadCubes(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
        prior=False,
        concat=True,
    ):
        super().__init__()
        self.concat = concat
        self.include_identity = network_config.include_identity
        if not prior:
            encoding = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    },
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    },
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    },
                    {
                        "n_dims_to_encode": 3,
                        **encoding_config.get_encoder_config()
                    }
                ]
            }
            if self.include_identity:
                encoding["nested"].append(
                    {
                        "n_dims_to_encode": 4,
                        "otype": "Identity"
                    }
                )
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=12 + (4 if self.include_identity else 0),
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        yxt = torch.cat([zyx[..., [1, 2]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        xzt = torch.cat([zyx[..., [2, 0]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        zyt = torch.cat([zyx[..., [0, 1]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        if self.include_identity:
            zyxt = torch.cat([zyx, torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
            inputs = torch.cat([zyx, yxt, xzt, zyt, zyxt], dim=-1)
        else:
            inputs = torch.cat([zyx, yxt, xzt, zyt], dim=-1)
        out = self.net(inputs)
        return out
    
class HyperCubes(nn.Module):
    def __init__(
        self,
        encoding_config: HashEncoderConfig,
        network_config: MLPNetConfig,
    ):
        super().__init__()
        self.include_identity = network_config.include_identity
        encoding = {
            "otype": "Composite",
            "nested": [
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 3,
                    **encoding_config.get_encoder_config()
                },
                {
                    "n_dims_to_encode": 4,
                    **encoding_config.get_encoder_config()
                }
            ]
        }
        if self.include_identity:
            encoding["nested"].append(
                {
                    "n_dims_to_encode": 4,
                    "otype": "Identity"
                }
            )
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=16 + (4 if self.include_identity else 0),
            n_output_dims=1,
            encoding_config=encoding,
            network_config=network_config.get_network_config(),
        )

    def forward(self, zyx, t):
        yxt = torch.cat([zyx[..., [1, 2]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        xzt = torch.cat([zyx[..., [2, 0]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        zyt = torch.cat([zyx[..., [0, 1]], torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        zyxt = torch.cat([zyx, torch.full((zyx.size(0), 1), t, device=zyx.device)], dim=1)
        if self.include_identity:
            inputs = torch.cat([zyx, yxt, xzt, zyt, zyxt, zyxt], dim=-1)
        else:
            inputs = torch.cat([zyx, yxt, xzt, zyt, zyxt], dim=-1)
        out = self.net(inputs)
        return out


class KPlanes(nn.Module):
    def __init__(self, encoding_config: KPlanesEncoderConfig, network_config: MLPNetConfig):
        super().__init__()
        self.multiscale_res_multipliers = [1, 2, 4, 8]
        self.concat_features = True
        self.encoding_config = encoding_config

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            # encoding_config = self.encoding_config.copy()  # Avoids in-place problems
            # Resolution fix: multi-res only on spatial planes
            resolution = [r * res for r in encoding_config.resolution[:3]] + encoding_config.resolution[3:]

            gp = init_grid_param(
                grid_nd=encoding_config.grid_dimensions,
                in_dim=encoding_config.input_coordinate_dim,
                out_dim=encoding_config.output_coordinate_dim,
                reso=resolution,
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)

            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config=network_config.get_network_config(),
            )

    def forward(self, x, t=None):
        if t is not None:
            x = torch.cat([x, torch.full((x.size(0), 1), t, device=x.device)], dim=1)

        features = interpolate_ms_features(
            x,
            ms_grids=self.grids,  # noqa
            grid_dimensions=self.encoding_config.grid_dimensions,
            concat_features=self.concat_features,
            num_levels=None,
        )

        output = self.sigma_net(features)
        return output
