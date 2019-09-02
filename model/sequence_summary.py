from keras.layers import Layer, Dense, Dropout
from keras import backend as K

import tensorflow as tf

class SequenceSummary(Layer):
    def __init__(self, name, n_embd=768):
        super(SequenceSummary, self).__init__(name=name)

        self.summary_type = 'cls_index'

        self.summary = Dense(
            units=1,
            input_shape=(n_embd,),
            name=name + '_dense'
        )

        self.first_dropout = Dropout(
            rate=0.1,
            name='_drop'
        )


    def call(self, inputs):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        hidden_states, cls_index = inputs
        # print(K.int_shape(hidden_states), "ZERO")
        # print(hidden_states, cls_index)
        # print(K.int_shape(cls_index), "ONE")
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
            # print(K.int_shape(output), "TWO")
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=torch.long)
            else:
                cls_index = K.expand_dims(K.expand_dims(cls_index, -1), -1)
                args = (1,) * (len(K.int_shape(cls_index)) - 1) + (K.int_shape(hidden_states)[-1],)
                cls_index = K.tile(cls_index, args)
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = K.squeeze(tf.gather(params=hidden_states, indices=tf.cast(cls_index, tf.int32), axis=-1), -2) # shape (bsz, XX, hidden_size)
        elif self.summary_type == 'attn':
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        # output = self.first_dropout(output)
        # output = self.summary(output)
        # output = self.activation(output)
        # output = self.last_dropout(output)

        return output
