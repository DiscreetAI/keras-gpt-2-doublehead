from keras import backend as K
from keras.metrics import top_k_categorical_accuracy
from tensorflow import one_hot

import tensorflow as tf

def cross_entropy(logits, labels, ignore_index=None):
    if ignore_index:
        print(K.int_shape(logits))
        labels = tf.cast(labels, tf.int32)
        # unc = tf.fill(tf.shape(labels), -1)
        # unc = K.not_equal(unc, labels)
        labels = K.reshape(tf.cast(one_hot(labels, 50257, axis=-1), tf.float32), (-1, 50257))
        xentropy = sigmoid_crossentropy_ignore_index(labels, logits)
    else:
        logits = K.reshape(logits, (1, -1))
        labels = K.reshape(labels, (1, -1))
        xentropy = K.mean(
                        K.categorical_crossentropy(
                            labels,
                            logits
                        )
                    )
    return xentropy
    
def mc_loss_function(mc_labels, mc_logits):
    mc_loss = cross_entropy( 
        K.reshape(mc_logits, (-1, K.int_shape(mc_logits)[-1])),
        mc_labels
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

def perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
#     cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.exp(K.mean(K.categorical_crossentropy(y_true, y_pred)))

def top_1(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def top_3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_positives = tf.cast(possible_positives, tf.float32)
    true_positives = tf.cast(true_positives, tf.float32)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    predicted_positives = tf.cast(predicted_positives, tf.float32)
    true_positives = tf.cast(true_positives, tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def perplexity_lm(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """

    # def sparse_crossentropy_ignore_index(y_true, y_pred):
    #     mul_1 = tf.multiply(
    #                 y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32)
    #             )
    #     temp = K.reshape(tf.cast(one_hot(tf.cast(y_true, tf.int32), 50257, axis=-1), tf.float32), (-1, 50257))     
    #     mul_2 = tf.multiply(
    #                 y_pred, tf.cast(tf.not_equal(temp, -1), tf.float32)
    #             )
    #     return K.mean(K.sparse_categorical_crossentropy(mul_1,mul_2), axis=-1)

    y_true = tf.cast(y_true, tf.int32)
    # # unc = tf.fill(tf.shape(labels), -1)
    # # unc = K.not_equal(unc, labels)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    # # y_pred = K.reshape(y_pred, (-1, 50257))
    # cross_entropy = sparse_crossentropy_ignore_index(y_true, y_pred)
    # perplexity = K.exp(cross_entropy)
    return perplexity(y_true, y_pred)

def perplexity_mc(y_true, y_pred):
    return K.exp(mc_loss_function(y_true, y_pred))

def top_1_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    y_pred = K.reshape(y_pred, (-1, 50257))
    return top_1(y_true, y_pred)

def top_1_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    y_pred = K.reshape(y_pred, (-1, 1))
    return top_1(y_true, y_pred)

def top_3_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (1, -1, 50257))
    return top_3(y_true[0], y_pred[0])

def top_3_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    return top_3(y_true, y_pred)

def precision_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    # y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    
    y_pred = K.argmax(y_pred, axis=-1)
    
    y_true = K.reshape(y_true, (1, -1))
    
    return precision_m(y_true, y_pred)

def precision_mc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    y_pred = K.reshape(y_pred, (1, -1))
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.reshape(y_true, (1,))
    return precision_m(y_true, y_pred)

def f1_score_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    # y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.reshape(y_true, (1, -1))
    return f1_m(y_true, y_pred)

def f1_score_mc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    y_pred = K.reshape(y_pred, (1, -1))
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.reshape(y_true, (1,))
    return f1_m(y_true, y_pred)

def get_metrics(is_mc=False):
    return [perplexity_mc] if is_mc else [perplexity_lm, precision_lm, f1_score_lm, top_1_lm]