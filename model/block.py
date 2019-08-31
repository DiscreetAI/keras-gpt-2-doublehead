from keras import backend as K
from keras.layers import Layer, Dropout
import keras
import numpy
import tensorflow as tf
from conv_1d import Conv1D
from mlp import MLP
from attention import Attention
from bert_layer_norm import BertLayerNorm as LayerNorm

class Block(Layer):
    def __init__(self, n_ctx, config, name, scale=False):
        super(Block, self).__init__(name=name)
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, name=name+"_layernorm1", eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, name="_attention", scale=scale)
        self.ln_2 = LayerNorm(nx, name=name+"_layernorm2", eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config, name="_mlp")
        self.name = name
        

    def call(self, x, layer_past=None, head_mask=None):
        output_attn = self.attn(self.ln_1(x), layer_past=layer_past, head_mask=head_mask)
        a = output_attn[0]

        x = x + a

        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = [h] + output_attn[1:]
        return outputs