import torch
import torch.nn as nn
import numpy as np

dim_s = 4
dim_c = 4
k = 3
n_fft_scale = {'bass': 8, 'drums': 2, 'other': 4, 'vocals': 3, '*': 2}


class Conv_TDF(nn.Module):
    def __init__(self, c, l, f, k, bn, bias=True):
        super(Conv_TDF, self).__init__()
        self.use_tdf = bn is not None
        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                )
            )
        if self.use_tdf:
            if bn == 0:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU()
                )
            else:
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    nn.BatchNorm2d(c),
                    nn.ReLU()
                )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x + self.tdf(x) if self.use_tdf else x


class Conv_TDF_net_trim(nn.Module):
    def __init__(self, L, l, g, dim_f, dim_t, target_name, k=3, hop=1024, bn=None, bias=True):
        super(Conv_TDF_net_trim, self).__init__()
        self.dim_f, self.dim_t = 2 ** dim_f, 2 ** dim_t
        self.n_fft = self.dim_f * n_fft_scale[target_name]
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.target_name = target_name
        out_c = dim_c
        in_c = dim_c
        self.n = L // 2

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=g, kernel_size=1, stride=1),
            nn.BatchNorm2d(g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.ds_dense = nn.ModuleList()
        self.ds = nn.ModuleList()
        for i in range(self.n):
            self.ds_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))
            scale = (2, 2)
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    nn.BatchNorm2d(c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.mid_dense = Conv_TDF(c, l, f, k, bn, bias=bias)

        self.us_dense = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            scale = (2, 2)
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    nn.BatchNorm2d(c - g),
                    nn.ReLU()
                )
            )
            f = f * 2
            c -= g
            self.us_dense.append(Conv_TDF(c, l, f, k, bn, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=out_c, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = x.transpose(-1, -2)
        ds_outputs = []
        for i in range(self.n):
            x = self.ds_dense[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)
        x = self.mid_dense(x)
        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.us_dense[i](x)
        x = x.transpose(-1, -2)
        x = self.final_conv(x)
        return x


class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        self.linear = nn.Linear((dim_s + 1) * 2, dim_s * 2, bias=False)

    def forward(self, x):
        x = x.reshape(1, (dim_s + 1) * 2, -1).transpose(-1, -2)
        x = self.linear(x)
        return x.transpose(-1, -2).reshape(dim_s, 2, -1)
