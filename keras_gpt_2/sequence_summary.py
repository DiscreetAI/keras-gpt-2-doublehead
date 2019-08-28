from keras.layers import Layer

class SequenceSummary(Layer):
    def __init__(self, config):
        super(SequenceSummary, self).__init__()

        self.summary_type = 'last'
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

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
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
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, XX, hidden_size)
        elif self.summary_type == 'attn':
            raise NotImplementedError

        # output = self.first_dropout(output)
        # output = self.summary(output)
        # output = self.activation(output)
        # output = self.last_dropout(output)

        return output
