import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from typing import Callable, Tuple
from modules.networks.transformer import TransformerEncoder, TransformerEncoderLayerResidual    
from modules.networks.s4 import S4Model, S4Model2D
import ccnn.models as models

def positional_encoding(x: torch.Tensor,
                        d_model:int,
                        seq_len: int):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, seq_len, d_model).to(x.device)
    pe[0, :, 0::2] = torch.sin(position * div_term).to(x.device)
    pe[0:, :, 1::2] = torch.cos(position * div_term).to(x.device)
    return x + pe


def get_encoder_1d(cfg: OmegaConf, 
                   d_input=1)-> Tuple[Callable, Callable]:
    if cfg.nn.actor_critic.encoder_net_1d == 's4':
        Encoder1D = S4Encoder(cfg, d_input)
    elif cfg.nn.actor_critic.encoder_net_1d == 'rnn':
        Encoder1D = RNNEncoder(cfg)
    elif cfg.nn.actor_critic.encoder_net_1d == 'ccnn':
        Encoder1D = CCNNEncoder1D(cfg)
    else:
        raise NotImplementedError
    return Encoder1D
    

def get_encoder_2d(cfg: OmegaConf)-> Tuple[Callable, Callable]:
    if cfg.nn.actor_critic.encoder_net_2d == 's4':
        Encoder2D = S4Encoder2D(cfg)
    elif cfg.nn.actor_critic.encoder_net_2d == 'cnn':
        Encoder2D = CNN(cfg)
    elif cfg.nn.actor_critic.encoder_net_2d == 'ccnn':
        Encoder2D = CCNNEncoder2D(cfg)
    elif cfg.nn.actor_critic.encoder_net_2d == 'resnet':
        Encoder2D = Resnet(cfg)
    else:
        raise NotImplementedError
    return Encoder2D


def get_decoder(cfg: OmegaConf)-> Tuple[Callable, Callable]:
    if cfg.nn.actor_critic.decoder_net == 's4':
        Decoder = S4Decoder(cfg)
        pass
    elif cfg.nn.actor_critic.decoder_net == 'rnn':
        Decoder = RNNDecoder(cfg)
    else:
        raise NotImplementedError
    return Decoder

    
def get_activation(activation_name: str):
    if activation_name == 'tanh':
        activation = nn.Tanh
    elif activation_name == 'learnable':
        pass
    elif activation_name == 'relu':
        activation = nn.ReLU
    elif activation_name == 'leakyrelu':
        activation = nn.LeakyReLU
    elif activation_name == "prelu":
        activation = nn.PReLU
    elif activation_name == 'gelu':
        activation = nn.GELU
    elif activation_name == 'sigmoid':
        activation = nn.Sigmoid
    elif activation_name in [ None, 'id', 'identity', 'linear', 'none' ]:
        activation = nn.Identity
    elif activation_name == 'elu':
        activation = nn.ELU
    elif activation_name in ['swish', 'silu']:
        activation = nn.SiLU
    elif activation_name == 'softplus':
        activation = nn.Softplus
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
    return activation


def layer_init(layer:torch.nn.Module,
               std=np.sqrt(2), 
               bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    # torch.nn.init.normal_(layer.weight, std)
    # torch.nn.init.normal_(layer.bias, std)
    return layer


class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.conv1 = layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4))
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        # self.bn3 = nn.BatchNorm2d(128)
        def conv2d_size_out(size, kernel_size, stride):
            # print((size - (kernel_size - 1) - 1) // stride + 1)
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(64, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)        
        convw = conv2d_size_out(convw, 3, 1)       
        
        self.linear_input_size = convw * convw * 64
        self.fc = layer_init(nn.Linear(self.linear_input_size, self.d_model))
        
    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # (batch_size, flaaten_size)
        x = self.fc(x)
        return x
    

class Resnet_Patch(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # TODO
    kernel_size = cfg.nn.cnn.kernel_size
    stride = cfg.nn.cnn.stride
    in_channels = 144
    mid_channels = cfg.nn.cnn.mid_channels
    final_channels = cfg.nn.actor_critic.d_model
    num_layers = cfg.nn.cnn.num_layers
    self.leakyrelu = nn.LeakyReLU()
    self.first_block = nn.Sequential(
          layer_init(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                            padding=1, bias=True)),
          nn.BatchNorm2d(mid_channels),
          nn.GELU())
    
    self.conv_layers = nn.ModuleList()
    self.channels = [mid_channels for i in range(num_layers)]
    for i in range(num_layers):  
        conv_block = nn.Sequential(
                                layer_init(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                padding=1, bias=True)),
                                nn.BatchNorm2d(mid_channels),
                                nn.GELU())
        self.conv_layers.append(conv_block)
    self.is_avg_pooling = cfg.nn.cnn.avg_pooling
    if self.is_avg_pooling:
        self.final_block = nn.Sequential(
              nn.Conv2d(mid_channels, final_channels, kernel_size=kernel_size, stride=stride,
                                padding=1, bias=True),
              nn.BatchNorm2d(final_channels),
              nn.GELU())
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
    else:
        input_size = mid_channels * 14 * 14
        self.fc = layer_init(nn.Linear(input_size, cfg.nn.actor_critic.d_model))

  def forward(self, x):
    if len(x.shape) < 4:
      x = x.unsqueeze(0)
    x = self.first_block(x)
    shortcut = x
    for conv_block in self.conv_layers:
      x = conv_block(x)
      x += shortcut
      shortcut = x
    if self.is_avg_pooling:
        x = self.final_block(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
    else:
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    return x


class Resnet(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # TODO
    kernel_size = cfg.nn.cnn.kernel_size
    stride = cfg.nn.cnn.stride
    in_channels = 4
    mid_channels = cfg.nn.cnn.mid_channels
    final_channels = cfg.nn.actor_critic.d_model
    num_layers = cfg.nn.cnn.num_layers
    self.leakyrelu = nn.LeakyReLU()
    self.first_block = nn.Sequential(
          layer_init(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                            padding=1, bias=True)),
          nn.BatchNorm2d(mid_channels),
          nn.GELU())
    
    self.conv_layers = nn.ModuleList()
    self.channels = [mid_channels for i in range(num_layers)]
    for i in range(num_layers):  
        conv_block = nn.Sequential(
                                layer_init(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                padding=1, bias=True)),
                                nn.BatchNorm2d(mid_channels),
                                nn.GELU())
        self.conv_layers.append(conv_block)
    self.is_avg_pooling = cfg.nn.cnn.avg_pooling
    if self.is_avg_pooling:
        self.final_block = nn.Sequential(
              nn.Conv2d(mid_channels, final_channels, kernel_size=kernel_size, stride=stride,
                                padding=1, bias=True),
              nn.BatchNorm2d(final_channels),
              nn.GELU())
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
    else:
        input_size = mid_channels * 64 * 64
        self.fc = layer_init(nn.Linear(input_size, cfg.nn.actor_critic.d_model))

  def forward(self, x):
    x = x.to(dtype=torch.float32) / 255.0
    if len(x.shape) < 4:
      x = x.unsqueeze(0)
    x = self.first_block(x)
    shortcut = x
    for conv_block in self.conv_layers:
      x = conv_block(x)
      x += shortcut
      shortcut = x
    if self.is_avg_pooling:
        x = self.final_block(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
    else:
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    return x



class RNNEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.final_act = cfg.nn.rnn.final_activation
        self.final_act_func = get_activation(self.final_act)()
        self.embedding = nn.Linear(1, self.d_model)
        self.num_layers = cfg.nn.rnn.num_layers
        self.bias = bool(cfg.nn.rnn.bias)
        self.rnn = nn.GRU(input_size=self.d_model,
                          hidden_size=self.d_model,
                          num_layers=self.num_layers,
                          bias=self.bias,
                          batch_first=True,
                          bidirectional=True
                          )
                                                
    def forward(self, x: torch.Tensor):
        #  # x: [batch_size, feature_dim]
        # print(x.storage())
        batch_size = x.size(0)
        feature_dim = x.size(1)
        ux = x.unsqueeze(-1) # x: [batch_size, feature_dim, 1]
        ux = self.embedding(ux) # ux: [batch_size, feature_dim, 32]
        weights, h_n = self.rnn(ux) # weights shape: [batch_size, feature_dim, 2*d_model]
        weights = weights.reshape(batch_size, feature_dim, 2, self.d_model)
        weights = weights.mean(dim=2, keepdim=False)
        weights = self.final_act_func(weights)
        return weights

    
class RNNDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.pos_encoding = cfg.nn.actor_critic.pos_encoding
        self.final_act = cfg.nn.rnn.final_activation
        self.final_act_func = get_activation(self.final_act)()
        self.num_layers = cfg.nn.rnn.num_layers
        self.bias = bool(cfg.nn.rnn.bias)
        # GRU input: (N,L,H_in) output: (N,L,H_out)
        self.rnn = nn.GRU(input_size=self.d_model, 
                          hidden_size=self.d_model,
                          num_layers=self.num_layers,
                          bias=self.bias,
                          batch_first=True,
                          bidirectional=True
                          )
        
    def forward(self, 
                out_dim: int, 
                embed_featrue: torch.Tensor):
        # shared_feature: [batch_size, shared_dim]
        batch_size, *dims = embed_featrue.shape
        x = embed_featrue.unsqueeze(1) 
        # embed_featrue: [batch_size, 1, shared_dim]
        # features = [embed_featrue for i in range(out_dim)]
        # embed_featrue = torch.cat(features, dim=1) 
        x = x.expand(batch_size, out_dim, *dims)
        x = x.reshape(batch_size, out_dim, -1)
         # embed_featrue: [batch_size, out_dim, d_model]
        if self.pos_encoding:
            x = positional_encoding(x, self.d_model, out_dim)
        weights, _ = self.rnn(x) 
        # weights shape: [batch_size, out_dim, 2*d_model]
        weights = weights.reshape(batch_size, out_dim, 2, self.d_model)
        weights = weights.mean(dim=2, keepdim=False)
        # weights shape: [batch_size, out_dim, d_model]
        weights = self.final_act_func(weights)
        return weights
    

class S4Encoder(nn.Module):
    def __init__(self, 
                 cfg: DictConfig, 
                 d_input=1):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.final_act = cfg.nn.s4.final_activation
        self.initializer = cfg.nn.s4.initializer
        self.s4_num_layers = cfg.nn.s4.num_layers
        self.s4_dropout = cfg.nn.s4.dropout
        self.s4_model = S4Model(
                        d_input=d_input,
                        d_model=self.d_model,
                        n_layers=self.s4_num_layers,
                        dropout=self.s4_dropout,
                        lr=cfg.nn.s4.lr,
                        )
                                                
    def forward(self, x:torch.Tensor):
        #  # x: [batch_size, feature_dim]
        ux = x.unsqueeze(-1) # ux: [batch_size, feature_dim, 1]

        weights = self.s4_model(ux) # weights shape: [batch_size, feature_dim, d_model]
        return weights

class S4Layer(nn.Module):
    def __init__(self, 
                 cfg: DictConfig, 
                 d_input=1):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.pos_encoding = cfg.nn.actor_critic.pos_encoding
        self.final_act = cfg.nn.s4.final_activation
        self.initializer = cfg.nn.s4.initializer
        self.s4_num_layers = cfg.nn.s4.num_layers
        self.s4_dropout = cfg.nn.s4.dropout
        self.s4_model = S4Model(
                        d_input=self.d_model,
                        d_model=self.d_model,
                        n_layers=self.s4_num_layers,
                        dropout=self.s4_dropout,
                        lr=cfg.nn.s4.lr,
                        )
                                                
    def forward(self, x:torch.Tensor):
        #  # x: [batch_size, seq_len, d_model]
        out = self.s4_model(x) # weights shape: [batch_size, feature_dim, d_model]
        return out

class S4Encoder2D(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.in_channels = 4
        self.d_state = cfg.nn.s4.d_state
        self.num_layers = cfg.nn.s4.num_layers
        self.dropout = cfg.nn.s4.dropout
        self.s4_2d = S4Model2D(
                    d_input=self.in_channels,
                    d_state=self.d_state,
                    d_model=self.d_model,
                    n_layers=self.num_layers,
                    dropout=self.dropout,
                    prenorm=False,
                    )
                                                
    def forward(self, x:torch.Tensor):
        #  # x: [batch_size, 4, W, H]
        x = self.s4_2d(x)
        return x


class S4Decoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.pos_encoding = cfg.nn.actor_critic.pos_encoding
        self.final_act = cfg.nn.s4.final_activation
        self.initializer = cfg.nn.s4.initializer
        self.s4_num_layers = cfg.nn.s4.num_layers
        self.s4_dropout = cfg.nn.s4.dropout
        self.s4_model = S4Model(
                        d_input=self.d_model,
                        d_model=self.d_model,
                        n_layers=self.s4_num_layers,
                        dropout=self.s4_dropout,
                        lr=cfg.nn.s4.lr,
                        )
        
    def forward(self,
                out_dim:int,
                embed_featrue:torch.Tensor):
        # shared_feature: [batch_size, d_model]
        batch_size = embed_featrue.size(0)
        x = embed_featrue.unsqueeze(1) 
        # x: [batch_size, 1, d_model]
        x = x.expand(batch_size, out_dim, self.d_model)
        if self.pos_encoding:
            x = positional_encoding(x, self.d_model, out_dim)
         # x: [batch_size, out_dim, d_model]
        weights = self.s4_model(x) 
        return weights


class CCNNEncoder1D(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.in_channels = 1
        self.mid_channels = self.d_model
        data_type = "sequence"
        net_type = f"{cfg.ccnn_seq.net.type}_{data_type}"
        net_type = getattr(models, net_type)
        self.ccnn_network = net_type(
            in_channels=self.in_channels,
            mid_channels=self.mid_channels if self.mid_channels != 2 else 1,
            net_cfg=cfg.ccnn_seq.net,
            kernel_cfg=cfg.ccnn_seq.kernel,
            conv_cfg=cfg.ccnn_seq.conv,
            mask_cfg=cfg.ccnn_seq.mask,
        )
              
    def forward(self, x):
        #  # x: [batch_size, feature_dim]
        ux = x.unsqueeze(1) # ux: [batch_size, feature_dim, 1]
        weights = self.ccnn_network(ux) # weights shape: [batch_size, d_model]
        return weights
    

class CCNNEncoder2D(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        self.in_channels = 4
        self.mid_channels = self.d_model
        data_type = "image"
        net_type = f"{cfg.ccnn_img.net.type}_{data_type}"
        net_type = getattr(models, net_type)
        self.ccnn_network = net_type(
            in_channels=self.in_channels,
            mid_channels=self.mid_channels if self.mid_channels != 2 else 1,
            net_cfg=cfg.ccnn_img.net,
            kernel_cfg=cfg.ccnn_img.kernel,
            conv_cfg=cfg.ccnn_img.conv,
            mask_cfg=cfg.ccnn_img.mask,
        )
                                                
    def forward(self, x):
        #  # x: [batch_size, 4, W, H]
        weights = self.ccnn_network(x) # weights shape: [batch_size, d_model]
        return weights