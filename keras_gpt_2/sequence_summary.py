from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

import tensorflow as tf

class SequenceSummary(Layer):
    def __init__(self, name):
        super(SequenceSummary, self).__init__(name=name)

        self.summary_type = 'cls_index'
        # self.summary = Identity()
        # if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
        #     if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
        #         num_classes = config.num_labels
        #     else:
        #         num_classes = config.hidden_size
        #     self.summary = nn.Linear(config.hidden_size, num_classes)

        # self.activation = Identity()
        # if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
        #     self.activation = nn.Tanh()

        # self.first_dropout = Identity()
        # if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
        #     self.first_dropout = nn.Dropout(config.summary_first_dropout)

        # self.last_dropout = Identity()
        # if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
        #     self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def call(self, inputs):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        hidden_states, cls_index = inputs

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
                print(cls_index.shape)                
                cls_index = K.expand_dims(cls_index, -1)
                print(cls_index.shape)
                args = (1,) * (len(K.int_shape(cls_index)) - 1) + (K.int_shape(hidden_states)[-1],)
                # cls_index = K.tile(cls_index, args)
                # cls_index = K.squeeze(cls_index, axis=-2)
                print(hidden_states.shape, "HIDDEN_STATES")
                print(cls_index.shape, "CLS_INDEX")
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output1, output2 = tf.split(hidden_states, 2)
            idx = tf.stack([tf.range(tf.shape(cls_index)[0]),cls_index[:,0]],axis=-1)
            output1 = tf.gather_nd(output1,idx)
            output2 = tf.gather_nd(output2,idx)
            final_gather = K.concatenate([output1, output2], axis=0)
            print(final_gather.shape)
            # gather_shape = tf.gather(params=hidden_states, indices=tf.cast(cls_index, tf.int32), axis=-1)
            # print(gather_shape.shape)
            return final_gather
            output = K.squeeze(tf.gather(params=hidden_states, indices=tf.cast(cls_index, tf.int32), axis=-2), -2) # shape (bsz, XX, hidden_size)
            print(output.shape, "SEQUENCE")
        elif self.summary_type == 'attn':
            raise NotImplementedError

        # output = self.first_dropout(output)
        # output = self.summary(output)
        # output = self.activation(output)
        # output = self.last_dropout(output)
        return output
