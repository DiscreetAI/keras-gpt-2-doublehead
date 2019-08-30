from keras import backend as K
from keras.layers import Layer, Dropout
import keras
import numpy
import tensorflow as tf
from conv_1d import Conv1D

class BertLayerNorm(Layer):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.variance_epsilon = eps

    def build():
        self.weight = self.add_weight(
            shape=(hidden_size,),
            initializer=keras.initializers.Ones()
        ) 
        self.bias = self.add_weight(
            shape=(hidden_size,),
            initializer=keras.initializers.Zeros()
        ) 
        

    def forward(self, x):
        u = K.mean(x, axis=-1, keepdim=True)
        s = K.mean(K.pow((x - u), 2), -1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
