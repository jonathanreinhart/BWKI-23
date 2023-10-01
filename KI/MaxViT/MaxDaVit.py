from typing import Tuple

import torch
import torch.nn as nn
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
import torchinfo
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange
import time
import numpy as np
import random
import sys

sys.path.append('KI/Dataset/')
from EEGDatasets import MotorImgDataset

"""changes to working version: pin_memory=true, num_workers=3, filepath: E:/..., benchmarking, grads to none, mixed precision, depths, AdamW"""

BATCH_SIZE = 32
VBATCH_SIZE = 4

# reproducibility
seed = 21
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        # self.dropsample = Dropsample(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fn(x)
        # out = self.dropsample(out)
        out = self.dropout(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

class RelPosBias(nn.Module):
    r"""Relative position bias module.
    Important: when only w1 is given, 1d embedding is used.
    """
    def __init__(self,dim,w1,w2 = 0):
        super().__init__()
        #do 2d embedding
        if(w2 != 0):
            self.rel_pos_bias = nn.Embedding((2 * w1 - 1)*(2 * w2 - 1), dim)

            pos1 = torch.arange(w1)
            pos2 = torch.arange(w2)
            grid = torch.stack(torch.meshgrid(pos1, pos2, indexing = 'ij'))
            #print(grid.shape)
            grid = rearrange(grid, 'c i j -> (i j) c')
            #print("grid: ", grid)

            #subtract each entry from each other entry to get distance
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            #print("rel_pos: ", rel_pos)
            rel_pos[:,:,0] += w1 - 1
            rel_pos[:,:,1] += w2 - 1

            #we want to a unique index between 0 and (2 * w1 - 1)*(2 * w2 - 1) for 
            #each pair, so we have to add 2wy-1 to the x difference
            rel_pos_indices = (rel_pos * torch.tensor([2 * w2 - 1, 1])).sum(dim = -1)
            #print("summed_ind: ", rel_pos_indices)
            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
        #just do 1d embedding
        else:
            self.rel_pos_bias = nn.Embedding(2 * w1 - 1, dim)
            pos = [torch.arange(0, (-1) * w1, -1) + i for i in range(w1)]
            pos = torch.stack(pos)
            rel_pos_indices = pos + w1 - 1
            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
        
    def forward(self,x):
        #print("ind_shape: ", self.rel_pos_indices.shape)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        bias = rearrange(bias, "w h c -> c w h")
        #print(f"bias shape: {bias.shape}")
        return x + bias

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 4,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        window_size = cast_tuple(window_size,2)
        assert len(window_size)<3
        w1,w2 = window_size

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        #relative positional bias
        self.rel_pos_bias = RelPosBias(self.heads,w1,w2)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flattent to 1d data with dimension d, so basically image is now like big time signal
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # simalarity matrix
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        #print("sim_shape before Pos: ", sim.shape)
        sim = self.rel_pos_bias(sim)
        #print("sim after Pos: ", sim)

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=False):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                            dim,
                            to_2tuple(k),
                            to_2tuple(1),
                            to_2tuple(k // 2),
                            groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x

class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        #print("num_heads, head_dim: ", self.num_heads, head_dim)
        #relative positional bias
        self.rel_pos_bias = RelPosBias(self.num_heads,head_dim)

    def forward(self, x):
        #signal is alreday flattened
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        
        '''changes: einsum and first mult q, k and not k, v'''
        #print("q-shape, k-shape: ", q.shape, k.shape)
        sim = einsum("b h p g, b h p c -> b h g c", q, k)
        #print("sim_shape before attn: ", sim.shape)
        sim = self.rel_pos_bias(sim)
        #print("sim after attn: ", sim)
        sim = sim.softmax(dim=-1)
        x = einsum("b h g c, b h p c -> b h g p", sim, v).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                ffn=True, cpe_act=False, input_shape=(160,64)):
        super().__init__()

        #relative positional bias
        #self.rel_pos_bias = RelPosBias(input_shape[0],input_shape[1],num_heads)

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                ConvPosEnc(dim=dim, k=3, act=cpe_act)])

        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x):
        r""" forward
        Args:
            x: input of shape (B,N,C)
        """
        size = (x.shape[2],x.shape[3])
        # flatten
        x = rearrange(x, 'b d w1 w2 -> b (w1 w2) d')
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, "b (w1 w2) d -> b d w1 w2", w1=size[0], w2=size[1])
        return x

class MaxDaViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_channel_head = 4,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        input_shape = (1, 160, 64)
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple of integers indicating number of transformer blocks at that stage'

        # convolutional stem
        dim_conv_stem = default(dim_conv_stem, dim)

        channels = input_shape[0]

        """!!!changed: stride = 2"""
        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride=1, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        window_size = cast_tuple(window_size,2)
        assert len(window_size)<3
        w1,w2 = window_size

        # iterate through stages
        #print("ind|stage_ind|stage_dim_in|stage_dim|layer_depth")
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                assert layer_dim%dim_channel_head == 0
                channel_head_num = layer_dim//dim_channel_head

                #print("{:>3}|{:>9}|{:>12}|{:>9}|{:>11}".format(ind,stage_ind,stage_dim_in,layer_dim,layer_depth))

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w1, w2 = w2),  # block-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w1, w2 = w2),  # grid-like attention
                    PreNormResidual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                    PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                    #ChannelBlock(layer_dim, channel_head_num)
                )
                self.layers.append(block)

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        #print(f"input_shape: {x.shape}")
        x = self.conv_stem(x)
        #print(f"conv_stem_out_shape: {x.shape}")
        for stage in self.layers:
            x = stage(x)
            #print("x_shape: ", x.shape)

        return self.mlp_head(x)

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    motImgDataset = MotorImgDataset("E:/Documents/datasets/MotorIMGTorch")
    training_loader, val_loader, test_loader = motImgDataset.get_train_val_test_dataloader([0.8,0.05,0.15], BATCH_SIZE, vbatch_size=VBATCH_SIZE, num_workers=3, pin_memory=True)

    maxDaVit = MaxDaViT(num_classes=5,dim=32,depth=(1,1,2,1),dropout=0.1,dim_conv_stem=3,window_size=(10,4))

    torchinfo.summary(maxDaVit.cuda(), input_size=(BATCH_SIZE, 1, 160, 64))

    #test MaxDaVit on dataset

    model = maxDaVit
    model.to(device)

    #weights to balance the classes
    weights = torch.tensor([0.25,1,1,1,1]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    """optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)"""
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=6.9e-5,eps=1e-2)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('tensorboard/runs/maxDaVit_trainer_{}'.format(timestamp))

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        last_time = time.time()
        correct = 0
        report_interval = 400

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair, loaded asynchronously
            inputs, labels = data
            inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
            
            # Zero gradients for every batch
            model.zero_grad(set_to_none=True)

            # """maybe add mixed precision"""
            # # Forward pass with mixed precision
            # with torch.cuda.amp.autocast(): # autocast as a context manager
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)


            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            running_loss += loss.item()
            if i % report_interval == report_interval-1:
                # torch.save(model.state_dict(), "AIModels/EEG/Trained models/MaxVit11.pt")
                last_loss = running_loss / report_interval # loss per batch

                # reproducibility
                seed = 21
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                vcorrect = 0
                running_vloss = 0.0
                
                model.eval()
                for j, vdata in enumerate(val_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                    with torch.no_grad():
                        voutputs = model(vinputs)
                        vloss = loss_fn(voutputs, vlabels)

                    vcorrect += (voutputs.argmax(1) == vlabels).type(torch.float).sum().item()
                    running_vloss += vloss
                    # just validate over 400 instances, to save time
                    if(j==400):
                        break

                model.train()

                avg_vloss = running_vloss / (j + 1)
                vcorrect = vcorrect / (j + 1) / VBATCH_SIZE

                print('  batch {} train-loss: {} train-acc: {} val-loss: {} val-acc: {} time: {}'.format(i + 1, last_loss, correct/BATCH_SIZE/report_interval, avg_vloss, vcorrect, time.time() - last_time))
                # print('  batch {} train-loss: {} train-acc: {} time: {}'.format(i + 1, last_loss, correct/BATCH_SIZE/report_interval, time.time() - last_time))
                last_time = time.time()

                # Log the running loss averaged per batch
                # for both training and validation
                tb_x = epoch_index * len(training_loader) + i + 1
                writer.add_scalars('loss', {
                    'train': last_loss,
                    'val': avg_vloss.item(),
                }, tb_x)
                writer.add_scalars('acc', {
                    'train': correct/BATCH_SIZE/report_interval,
                    'val': vcorrect,
                }, tb_x)
                writer.flush()
                running_loss = 0.
                correct = 0

        return last_loss

    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    torch.backends.cudnn.benchmark = True
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(epoch_number, writer)

        """
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print('LOSS train {} val {}'.format(avg_loss,avg_loss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training Loss',
                        { 'Training' : avg_loss},
                        epoch_number + 1)
        writer.flush()
        """
        epoch_number += 1

    writer.close()
if __name__ == "__main__":
    main()