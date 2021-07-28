import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from cyclegan.networks import get_norm_layer, init_net


class FeatureMapDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=2048, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(FeatureMapDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class FeatureMapGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=512, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(FeatureMapGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
    
        if n_blocks == 0:
            model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]
            model += [nn.Conv2d(ngf, ngf // 2, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf // 2),
                    nn.ReLU(True)]
            model += [nn.ConvTranspose2d(ngf // 2, ngf, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]
            model += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=3, padding=1, bias=use_bias),
                    nn.ReLU(True)]
        else:
            model = [nn.ReflectionPad2d(1),
                    nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]
            n_downsampling = 1
            for i in range(n_downsampling):  # add downsampling layers
                mult = 2 ** i
                model += [nn.Conv2d(int(ngf / mult), int(ngf / mult / 2), kernel_size=3, stride=2, padding=1, bias=use_bias),
                        norm_layer(int(ngf / mult / 2)),
                        nn.ReLU(True)]

            mult = 2 ** n_downsampling
            
            for i in range(n_blocks):       # add ResNet blocks
                model += [ResnetBlock(int(ngf / mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

            for i in range(n_downsampling):  # add upsampling layers
                mult = 2 ** (n_downsampling - i)
                model += [nn.ConvTranspose2d(int(ngf / mult), int(ngf / mult * 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=0,
                                            bias=use_bias),
                        norm_layer(int(ngf / mult * 2)),
                        nn.ReLU(True)]
            
            model += [nn.ReflectionPad2d(1)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
            model += [nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class FeatureMapGeneratorPanel(nn.Module):
    def __init__(self, nc, ngf, norm, use_dropout, n_experts, init_type, init_gain, gpu_ids, n_blocks=1):
        super(FeatureMapGeneratorPanel, self).__init__()
        net_list = nn.ModuleList()
        norm_layer = get_norm_layer(norm_type=norm)
        for i in range(0, n_experts):
            net = FeatureMapGenerator(nc, nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
            net = init_net(net, init_type, init_gain, gpu_ids)
            net_list.append(net)
        self.net_list = net_list
        self.n_experts = n_experts

    def forward(self, input):
        outputs = []
        for i in range(0, self.n_experts):
            outputs.append(self.net_list[i](input[i]))
        return outputs

    def get_expert(self, idx):
        return self.net_list[idx]