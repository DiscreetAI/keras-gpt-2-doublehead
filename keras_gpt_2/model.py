from .backend import keras
from keras_embed_sim import EmbeddingRet, EmbeddingSim
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_transformer import gelu, attention_builder, feed_forward_builder
from keras_transformer import get_custom_objects as get_transformer_custom_objects

from keras.layers import Dense, Layer
from keras import backend as K
from .sequence_summary import SequenceSummary

import tensorflow as tf
import numpy as np

__all__ = ['get_model', 'get_custom_objects']


def _wrap_layer(name, input_layer, build_func, trainable=True):
    """Wrap layers with normalization and residual.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(input_layer)
    build_output = build_func(normal_layer)
    return keras.layers.Add(name='%s-Add' % name)([input_layer, build_output])


def _get_encoder_component(name,
                           input_layer,
                           head_num,
                           hidden_dim,
                           attention_activation=None,
                           feed_forward_activation='relu',
                           trainable=True):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadAtt' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=True,
            trainable=trainable,
        ),
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        trainable=trainable,
    )
    return feed_forward_layer


def get_model(n_vocab,
              n_ctx=1024,
              n_embd=768,
              n_head=12,
              n_layer=12,
              batch_size=None,
              fixed_input_shape=False):
    """Get basic GPT-2 model.

    :param n_vocab: Number of vocabulary tokens.
    :param n_ctx: The length of each input.
    :param n_embd: The dimension of embeddings.
    :param n_head: Number of heads in transformer.
    :param n_layer: Number of transformer blocks.
    :param batch_size: Batch size of the model.
    :param fixed_input_shape: Whether the length of input is fixed. (Needed for TPU training)
    :return: The model.
    """
    if fixed_input_shape:
        input_layer_shape = (batch_size, n_ctx)
    else:
        input_layer_shape = (batch_size, None)
    input_layer = keras.layers.Input(
        batch_shape=input_layer_shape,
        name='Input',
    )

    embed_token, embeddings = EmbeddingRet(
        input_dim=n_vocab,
        output_dim=n_embd,
        mask_zero=False,
        name='Embed-Token',
    )(input_layer)
    embed_token_pos = PositionEmbedding(
        input_dim=n_ctx,
        output_dim=n_embd,
        mode=PositionEmbedding.MODE_ADD,
        name='Embed-Token-Pos',
    )(embed_token)

    last_layer = embed_token_pos
    for i in range(n_layer):
        last_layer = _get_encoder_component(
            name='Encode-%d' % i,
            input_layer=last_layer,
            head_num=n_head,
            hidden_dim=n_embd * 4,
            attention_activation=None,
            feed_forward_activation=gelu,
        )

    norm_layer = LayerNormalization(
        name='Norm',
    )(last_layer)

    lm_head = EmbeddingSim(
        use_bias=False,
        name='LMOutput',
    )([norm_layer, embeddings])

    mc_head = SequenceSummary(
        name='MCOutput'
    )(norm_layer)

    # output_layer = 

    losses = {
        "LMOutput": lm_loss_function,
        "MCOutput": mc_loss_function,
    }
    lossWeights = {"LMOutput": 2.0, "MCOutput": 1.0}

    model = keras.models.Model(inputs=input_layer, outputs=[lm_head, mc_head])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=losses,
        loss_weights=lossWeights
    )
    return model

def cross_entropy(logits, labels, ignore_index=None):
    if ignore_index:
        unc = [0 if i == ignore_index else 1 for i in range(50257)]
        unc = tf.convert_to_tensor(unc)
        xentropy = tf.reduce_mean(
            tf.losses.compute_weighted_loss(
                weights = tf.cast(unc, tf.float32),
                losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = logits,
                    labels = tf.cast(labels, tf.float32)
                )
            ), 
            name='xentropy'
        )
    else:
        xentropy = tf.reduce_mean(
            tf.losses.compute_weighted_loss(
                losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = logits,
                    labels = tf.cast(labels, tf.float32)
                )
            ), 
            name='xentropy'
        )
    return xentropy
    
def mc_loss_function(mc_labels, mc_logits):
    mc_loss = cross_entropy( 
        K.reshape(mc_logits, (-1, K.int_shape(mc_logits)[-1])),
        K.flatten(mc_labels)
    )

    return mc_loss

def lm_loss_function(lm_labels, lm_logits):
    shift_logits = lm_logits[..., :-1, :]
    shift_labels = lm_labels[..., 1:]
    lm_loss = cross_entropy(
        K.reshape(shift_logits, (-1, K.int_shape(shift_logits)[-1])),
        K.flatten(shift_labels),
        -1
    )

    return lm_loss



    

def get_custom_objects():
    custom_objects = get_transformer_custom_objects()
    custom_objects['gelu'] = gelu
    custom_objects['PositionEmbedding'] = PositionEmbedding
    return custom_objects
