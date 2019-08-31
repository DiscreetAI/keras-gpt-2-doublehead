from keras import backend as K
from keras.layers import Layer, Dropout
import keras
import numpy
import tensorflow as tf
from conv_1d import Conv1D

class BertLayerNorm(Layer):
    def __init__(self, hidden_size, name, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__(name=name)
        self.variance_epsilon = eps
        self.name = name
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.hidden_size,),
            initializer=keras.initializers.Ones(),
            name=self.name + "_weight"

        ) 
        self.bias = self.add_weight(
            shape=(hidden_size,),
            initializer=keras.initializers.Zeros(),
            name=self.name + "_bias"
        ) 

        super(BertLayerNorm, self).build(input_shape)
        

    def call(self, x):
        u = K.mean(x, axis=-1, keepdim=True)
        s = K.mean(K.pow((x - u), 2), -1, keepdim=True)
        x = (x - u) / K.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
