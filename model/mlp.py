from keras import backend as K
from keras.layers import Layer, Dropout
import keras
import numpy
import tensorflow as tf
from conv_1d import Conv1D

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MLP(Layer):
    def __init__(self, n_state, config, name):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx, name=name+"_conv1")
        self.c_proj = Conv1D(nx, n_state, name=name+"_conv1")
        self.act = gelu
        self.dropout = Dropout(config.resid_pdrop, name=name+"_drop")
        self.name = name

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)