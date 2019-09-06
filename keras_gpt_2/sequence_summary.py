from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

import tensorflow as tf

class SequenceSummary(Layer):
    def __init__(self, name):
        super(SequenceSummary, self).__init__(name=name)

        self.summary_type = 'last'

    def call(self, inputs):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        hidden_states = inputs

        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=torch.long)
            else:
                cls_index = K.expand_dims(cls_index, -1)
                args = (1,) * (len(K.int_shape(cls_index)) - 1) + (K.int_shape(hidden_states)[-1],)
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            cls_index = tf.cast(cls_index, tf.int32)
            intended_shape = (tf.shape(cls_index)[0],)
            idx = tf.stack([tf.range(tf.shape(cls_index)[0]), K.reshape(cls_index[:,0], intended_shape)],axis=-1)
            output = tf.gather_nd(hidden_states, idx)
        elif self.summary_type == 'attn':
            raise NotImplementedError

        return output
