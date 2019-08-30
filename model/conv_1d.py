from keras import backend as K
from keras.layers import Layer
import numpy
import tensorflow as tf
import keras

class Conv1D(Layer):
    def __init__(self, nf, nx, name):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__(name=name)
        self.nf = nf
        self.nx = nx
        self.bias = None
        self.weight = None
        self.name = name
       

    def build(self):
        w = K.random_normal(shape=(self.nx, self.nf), stddev=0.02)
        self.weight = self.add_weight(
            shape=(self.nx, self.nf),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name=self.name + "_weight"
        )
        self.bias = self.add_weight(
            shape=(self.nf,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name=self.name + "_bias"
        )

    def call(self, x):
        size_out = K.int_shape(x)[:-1] + (self.nf,)
        x = self.bias + K.dot(K.reshape(x, (-1, K.int_shape(x)[-1])), self.weight)
        x = K.reshape(x, *size_out)
        return x