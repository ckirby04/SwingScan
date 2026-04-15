"""Vendored SwingNet architecture (MobileNetV2 + BiLSTM).

This file contains an adaptation of the SwingNet model architecture
from:

    wmcnally/golfdb  (https://github.com/wmcnally/golfdb)
    CVPR 2019 Workshop paper:
        "GolfDB: A Video Database for Golf Swing Sequencing"
        McNally et al., CVPR 2019 Workshops.

The upstream project is released under a Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
The MobileNetV2 block implementation is further credited to:

    tonylins/pytorch-mobilenet-v2
        (https://github.com/tonylins/pytorch-mobilenet-v2)

SwingScan vendors this architecture so we can load the upstream's
pretrained ``swingnet_1800.pth.tar`` checkpoint without a runtime
dependency on the upstream repo, and so we can run inference on CPU
and Python 3.11. Adaptations from the upstream code:

- Removed hard-coded ``.cuda()`` calls in ``init_hidden``; the hidden
  state is allocated on whichever device the model was moved to.
- Removed the ``torch.autograd.Variable`` wrapper (Variable has been
  a no-op alias for Tensor since PyTorch 0.4).
- Dropped the ``state_dict_mobilenet`` eager load from the ctor:
  SwingNet's checkpoint already contains the MobileNetV2 weights, so
  initializing from ImageNet pretrain is unnecessary and would
  actually break inference when the pretrain file is absent.

Any non-commercial use of this module must carry this attribution.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

__all__ = ["EventDetector", "load_swingnet_checkpoint"]


# ---------------------------------------------------------------------------
# MobileNetV2 backbone (unchanged from upstream, minus pretrain autoloading).
# ---------------------------------------------------------------------------


def _conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def _conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class _InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.conv(x) if self.use_res_connect else self.conv(x)
        return result


class _MobileNetV2(nn.Module):
    def __init__(self, n_class: int = 1000, input_size: int = 224, width_mult: float = 1.0) -> None:
        super().__init__()
        block = _InvertedResidual
        min_depth = 16
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting: list[list[int]] = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult) if width_mult >= 1.0 else input_channel
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        features: list[nn.Module] = [_conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = max(int(c * width_mult), min_depth)
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        features.append(_conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(3).mean(2)
        out: torch.Tensor = self.classifier(x)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ---------------------------------------------------------------------------
# EventDetector (SwingNet itself — MobileNetV2 backbone + BiLSTM + Linear).
# ---------------------------------------------------------------------------


class EventDetector(nn.Module):
    """BiLSTM over MobileNetV2 features, 9-way classifier.

    Output class layout (matches upstream training):

        0: Address
        1: Toe-up
        2: Mid-backswing (arm parallel)
        3: Top
        4: Mid-downswing (arm parallel)
        5: Impact
        6: Mid-follow-through (shaft parallel)
        7: Finish
        8: "no event"
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        lstm_layers: int = 1,
        lstm_hidden: int = 256,
        bidirectional: bool = True,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        net = _MobileNetV2(width_mult=width_mult)
        # Features 0..18 inclusive (the first 19 Sequential children) —
        # matches upstream slicing: `list(net.children())[0][:19]`.
        self.cnn = nn.Sequential(*list(net.features.children())[:19])

        feature_dim = int(1280 * width_mult if width_mult > 1.0 else 1280)
        self.rnn = nn.LSTM(
            feature_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if bidirectional:
            self.lin = nn.Linear(2 * lstm_hidden, 9)
        else:
            self.lin = nn.Linear(lstm_hidden, 9)

        if dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_directions = 2 if self.bidirectional else 1
        return (
            torch.zeros(
                num_directions * self.lstm_layers, batch_size, self.lstm_hidden, device=device
            ),
            torch.zeros(
                num_directions * self.lstm_layers, batch_size, self.lstm_hidden, device=device
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, C, H, W = x.size()
        hidden = self.init_hidden(batch_size, x.device)

        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in, hidden)
        out: torch.Tensor = self.lin(r_out)
        return out.view(batch_size * timesteps, 9)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def load_swingnet_checkpoint(weights_path: str, device: torch.device) -> EventDetector:
    """Construct an :class:`EventDetector` and load a SwingNet checkpoint.

    Args:
        weights_path: Path to ``swingnet_1800.pth.tar`` (or compatible).
        device: Target device (``torch.device('cpu')`` or ``'cuda'``).

    Returns:
        The model in eval mode, on the requested device.
    """
    model = EventDetector(
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )
    checkpoint: dict[str, Any] = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
