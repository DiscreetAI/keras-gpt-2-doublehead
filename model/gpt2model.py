from keras import backend as K
from keras.layers import Layer, Dropout, Embedding
import keras
import numpy as np
import tensorflow as tf
from block import Block
from bert_layer_norm import BertLayerNorm as LayerNorm

class GPT2Model(Layer):
    def __init__(self, config, name):
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = Embedding(config.vocab_size, config.n_embd, name=name+"_wte")
        self.wpe = Embedding(config.n_positions, config.n_embd, name=name+"_wpe")
        self.drop = Dropout(config.embd_pdrop, name=name+"_drop")
        self.block_list = [Block(config.n_ctx, config, name=name+"_block{}".format(i), scale=True) for i in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, name=name+"_layernorm", eps=config.layer_norm_epsilon)

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens):
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.input_dim, old_embeddings.output_dim
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        new_embeddings = Embedding(new_num_tokens, old_embedding_dim, name=old_embeddings.name)
        old_weight = old_embeddings.get_weights()[0]
        new_weight = new_embeddings.get_weights()[0]
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_weight[:num_tokens_to_copy, :] = old_weight[:num_tokens_to_copy, :]
        new_embeddings.set_weights([new_weight])

        return new_embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def call(input_ids, position_ids=None, token_type_ids=None, past=None, head_mask=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = K.int_shape(past[0][0], -2)

        if position_ids is None:
            position_ids = np.arange(past_length, K.int_shape(input_ids)[-1] + past_length)
            position_ids = tf.convert_to_tensor(position_ids, tf.int64)
            K.reshape(K.expand_dims(position_ids, 0), input_ids.shape)

        assert head_mask is None
        head_mask = [None] * self.config.n_layer

        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is None:
            token_type_embeds = 0
        else:
            token_type_ids = K.reshape(token_type_ids, (-1, K.int_shape(token_type_ids)[-1]))
            token_type_embeds = self.wte(token_type_ids)
        
        hidden_states = input_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (K.int_shape(hidden_states)[-1],)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (K.reshape(hidden_states, output_shape))

            outputs = block(hidden_states, layer_past, head_mask[i])
            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = K.reshape(hidden_states, output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + K.int_shape(all_attentions[0])[-2:]
            all_attentions = tuple(K.reshape(t, attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)

