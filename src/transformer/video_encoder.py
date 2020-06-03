import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models import shufflenet_v2_x1_0 as shufflenet
from torchvision import transforms
from attention import MultiHeadAttention
from module import PositionalEncoding, PositionwiseFeedForward
from utils import get_non_pad_mask, get_attn_pad_mask
from IPython import embed


class VideoEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, pe_maxlen=5000):
        super(VideoEncoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        net = shufflenet(pretrained=True)
        self.feature_extractor = nn.Sequential(net.conv1, net.maxpool, net.stage2, net.stage3, net.stage4)

        self.fc = nn.Linear(464 * 2 * 3, d_model)
        
        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            VideoEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])


    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        enc_slf_attn_list = []

        batch_ = padded_input.size(0)
        length = padded_input.size(1)
        video_feature_arr = []
        for i in range(length):
            ## todo: add image normalization
            frame_feature = self.fc(self.feature_extractor(padded_input[:,i,:,:,:]).view(batch_, -1))
            video_feature_arr.append(frame_feature)

        video_features = torch.stack(video_feature_arr, dim=1)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(video_features, input_lengths=input_lengths)
        slf_attn_mask = get_attn_pad_mask(video_features, input_lengths, length)

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(video_features) +
            self.positional_encoding(padded_input))

        #embed()
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class VideoEncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(VideoEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
