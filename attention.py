from keras import backend as K
from keras.layers import Layer, Dropout
import keras
import numpy
import tensorflow as tf
from conv_1d import Conv1D

class Attention(Layer):

    def __init__(self, nx, n_ctx, config, scale=False,  **kwargs):
        super(Attention, self).__init__(**kwargs)
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.bias = 
        #self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.output_attentions = config.output_attentions

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = Dropout(config.attn_pdrop)
        self.resid_dropout = Dropout(config.resid_pdrop)

    def build():
        def tril_initializer(shape):
            return K.reshape(tf.convert_to_tensor(np.tril(np.ones(shape))), (1, 1, n_ctx, n_ctx))

        self.add_weight(
            shape=(n_ctx, n_ctx),
            initializer=tril_initializer,
            trainable=False
        )

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     mask = keras.initializers.Ones(self.n_head, self.split_size // self.n_head)
    #     for head in heads:
    #         mask[head] = 0
    #     mask = mask.view(-1).contiguous().eq(1)
    #     index = torch.arange(len(mask))[mask].long()
    #     index_attn = torch.cat([index, index + self.split_size, index + (2*self.split_size)])
    #     # Prune conv1d layers
    #     self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
    #     self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
    #     # Update hyper params
    #     self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
    #     self.n_head = self.n_head - len(heads)

    def _attn(self, q, k, v, head_mask=None):
        w =  K.dot(q, k)
        if self.scale:
            w = w / math.sqrt(K.int_shape(v)[-1])
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : K.int_shape(w)[-2], : K.int_shape(w)[-1]]
        w = w * b + -1e9 * (1 - b)

        w = keras.activations.softmax(w, axis=-1)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [K.dot(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        new_x_shape = K.int_shape(x)[:-2] + (K.int_shape(x)[-2] * K.int_shape(x)[-1],)
        return K.reshape(x, new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = K.int_shape(x)[:-1] + (self.n_head, K.int_shape(x)[-1] // self.n_head)
        x = K.reshape(x, new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return K.permute_dimensions(x, (0, 2, 3, 1))
        else:
            return K.permute_dimensions(x, (0, 2, 1, 3))

    def forward(self, x, head_mask=None):
        x = self.c_attn(x)
        query, key, value = tf.split(x, self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)
        

