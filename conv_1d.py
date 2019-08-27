from keras import backend as K
from keras.layers import Layer
import numpy
import tensorflow as tf

class Conv1D(Layer):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        self.bias = None
        self.weight = None
       

    def build():
        w = K.random_normal(shape=(nx, nf), stddev=0.02)
        self.weight = self.add_weight(
            shape=(nx, nf),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(nf,),
            initializer=keras.initializers.Zeros()
            trainable=True
        )

    def forward(self, x):
        size_out = K.int_shape(x)[:-1] + (self.nf,)
        x = self.bias + K.dot(K.reshape(x, (-1, K.int_shape(x)[-1])), self.weight)
        x = K.reshape(x, *size_out)
        return x