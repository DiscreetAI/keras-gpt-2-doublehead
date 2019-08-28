from keras.layers import Dense, Layer
from keras import backend as K
from sequence_summary import SequenceSummary

class GPT2DoubleHeadsModel(Layer):
    def __init__(self, gpt2_model, config):
        self.transformer = gpt2_model
        self.lm_head = Dense(config.n_vocab, input_shape=(config.n_embd))
        self.multiple_choice_head = SequenceSummary(config)

    def call(self, transformer_outputs, mc_token_ids=None):
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        mc_logits = K.squeeze(self.multiple_choice_head(hidden_states, mc_token_ids), -1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]

        return outputs
