from gpt2model import GPT2Model
from keras.layers import Dense, Layer
from keras import backend as K
from sequence_summary import SequenceSummary

class GPT2DoubleHeadsModel(Layer):
    def __init__(self, config, name="gpt2doublesheadsmodel"):
        super(GPT2DoubleHeadsModel, self).__init__()
        self.transformer = GPT2Model(config, name='gpt2model')
        self.lm_head = Dense(config.vocab_size, input_shape=(config.n_embd,), use_bias=False)
        self.multiple_choice_head = SequenceSummary(name=name+"_sequencesummary")

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self.lm_head.set_weights([self.transformer.wte])

    def call(self, input_ids, mc_token_ids=None, token_type_ids=None, position_ids=None, past=None, head_mask=None):
        transformer_outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                               past=past, head_mask=head_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = K.squeeze(self.multiple_choice_head(hidden_states, mc_token_ids), -1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        # if mc_labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)),
        #                     mc_labels.view(-1))
        #     outputs = (loss,) + outputs
        # if lm_labels is not None:
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = lm_labels[..., 1:].contiguous()
        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
        #                     shift_labels.view(-1))
        #     outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)