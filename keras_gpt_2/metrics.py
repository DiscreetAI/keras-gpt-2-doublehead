import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow import one_hot
from tensorflow.keras.callbacks import Callback

def perplexity(y_true, y_pred):
#     """
#     The perplexity metric. Why isn't this part of Keras yet?!
#     https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
#     https://github.com/keras-team/keras/issues/8267
#     """
# #     cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
#     return K.exp(K.mean(K.categorical_crossentropy(y_true, y_pred)))
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return K.mean(perplexity)

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
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), y_pred.shape)
    # # y_pred = K.reshape(y_pred, (-1, 50257))
    # cross_entropy = sparse_crossentropy_ignore_index(y_true, y_pred)
    # perplexity = K.exp(cross_entropy)
    return perplexity(y_true, y_pred)

def perplexity_mc(y_true, y_pred):
    cross_entropy = K.mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=K.reshape(tf.cast(y_true, tf.float32), (-1, 1)),
                            logits=K.reshape(y_pred, (-1, 1))
                        )
                    )
    
    return K.exp(cross_entropy)

def top_1_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    
    y_pred = K.reshape(y_pred, (-1, 50257))
    return K.mean(top_1(y_true, y_pred))

def top_1_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    y_pred = K.reshape(y_pred, (-1, 1))
    #print(y_true.shape, y_pred.shape, "HERE")
    return top_1(y_true, y_pred)

def top_3_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    y_pred = K.reshape(y_pred, (-1, 50257))
    return top_3(y_true, y_pred)

def top_3_mc(y_true, y_pred):
    y_true = K.reshape(tf.cast(y_true, tf.float32), (-1, 1))
    y_pred = K.reshape(y_pred, (-1, 1))
    return top_3(y_true, y_pred)

def precision_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    # y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    
    y_pred = K.argmax(y_pred, axis=-1)
    
    # y_true = K.reshape(y_true, (2, -1))
    
    return precision_m(y_true, y_pred)

def precision_mc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    y_pred = K.argmax(y_pred, axis=-1)
    #y_true = K.reshape(y_true, (2,))
    return precision_m(y_true, y_pred)

def f1_score_lm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    # y_true = K.reshape(tf.cast(one_hot(y_true, 50257, axis=-1), tf.float32), (-1, 50257))
    y_pred = K.argmax(y_pred, axis=-1)
    # y_true = K.reshape(y_true, (2, -1))
    return f1_m(y_true, y_pred)

def f1_score_mc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    # y_pred = K.reshape(y_pred, (2, -1))
    y_pred = K.argmax(y_pred, axis=-1)
    # y_true = K.reshape(y_true, (2,))
    return f1_m(y_true, y_pred)

class Metrics(Callback):
    def __init__(self, input_ids, lm_labels, mc_token_ids, mc_labels):
        self.input_ids = input_ids
        self.lm_labels = lm_labels
        self.mc_token_ids = mc_token_ids
        self.mc_labels = mc_labels

    def on_train_begin(self, logs={}):
        self.batch_loss = []
        self.batch_lm_loss = []
        self.batch_mc_loss = []

        lm_name = "LMOutput"
        mc_name = "MCOutput"
        endings = ["_perplexity", "_precision", "_top_1", "_top_3", "_f1_score"]
        names = [lm_name + ending for ending in endings] + [mc_name + ending for ending in endings]
        self.metrics = {name:[] for name in names}
        self.metrics['loss'] = []
        self.metrics['LMOutput_loss'] = []
        self.metrics['MCOutput_loss'] = []
        functions = [perplexity_lm, precision_lm, top_1_lm, top_3_lm, f1_score_lm, perplexity_mc, precision_mc, top_1_mc, top_3_mc, f1_score_mc]
        self.functions = dict(zip(names, functions))
        print("Initialized metrics!")
        print(self.metrics)
        print(self.functions)

    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))
        self.batch_lm_loss.append(logs.get('LMOutput_loss'))
        self.batch_mc_loss.append(logs.get('MCOutput_loss'))

    def on_epoch_end(self, epoch, logs={}):
        lm_logits, mc_logits = self.model.predict([self.input_ids, self.mc_token_ids])
        print(lm_logits.shape)
        print(mc_logits.shape)
        for name, function in self.functions.items():
            if name[:2] == 'LM':
                metric = function(tf.convert_to_tensor(self.lm_labels), tf.convert_to_tensor(lm_logits))
                self.metrics[name].append(metric)
                
            else:
                metric = function(tf.convert_to_tensor(self.mc_labels), tf.convert_to_tensor(mc_logits))
                self.metrics[name].append(metric)
            print(name, metric)
        self.metrics['loss'].append(self.batch_loss)
        self.metrics['LMOutput_loss'].append(self.batch_lm_loss)
        self.metrics['MCOutput_loss'].append(self.batch_mc_loss)
        self.batch_loss = []
        self.batch_lm_loss = []
        self.batch_mc_loss = []
            

def get_metrics(is_mc=False):
    return [perplexity_mc, precision_mc, f1_score_mc, top_1_mc, top_3_mc] if is_mc else [perplexity_lm, precision_lm, f1_score_lm, top_1_lm, top_3_lm]
