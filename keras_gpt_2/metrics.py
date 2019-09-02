from keras import backend as K
from keras.metrics import top_k_categorical_accuracy
from tensorflow import one_hot

import tensorflow as tf

def perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = K.exp(cross_entropy)
    return perplexity

def top_1(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def top_3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
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

    def sparse_crossentropy_ignore_index(y_true, y_pred):
        return K.mean(
            K.sparse_categorical_crossentropy(
                tf.multiply(
                    y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32)
                ),
                tf.multiply(
                    y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32)
                ),
                from_logits=True
            ), axis=-1)

    y_true = tf.cast(y_true, tf.int32)
    # unc = tf.fill(tf.shape(labels), -1)
    # unc = K.not_equal(unc, labels)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    y_pred = K.reshape(y_pred, (-1, 50257))
    cross_entropy = sparse_crossentropy_ignore_index(y_true, y_pred)
    perplexity = K.exp(cross_entropy)
    return perplexity

def perplexity_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    return perplexity(y_true, y_pred)

def top_1_lm(y_true, y_pred):
    print(y_pred.shape, y_true.shape)
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (1, -1, 50257))
    print(y_pred.shape, y_true.shape)
    return top_1(y_true[0], y_pred[0])

def top_1_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    return top_1(y_true, y_pred)

def top_3_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (1, -1, 50257))
    return top_3(y_true[0], y_pred[0])

def top_3_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    return top_3(y_true, y_pred)

def precision_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    return precision_m(y_true, y_pred)

def precision_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    return precision_m(y_true, y_pred)

def f1_score_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    return f1_m(y_true, y_pred)

def f1_score_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    return f1_m(y_true, y_pred)

def get_metrics(is_mc=False):
    return [perplexity_mc] if is_mc else [perplexity_lm]