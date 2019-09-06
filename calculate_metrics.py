import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import BaseLogger, History
import tensorflow as tf
import numpy as np
from collections import defaultdict
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
import requests
import gpt_2_simple as gpt2
from keras_gpt_2 import perplexity_lm, f1_score_lm, top_1_mc
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
import time

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 

model_folder = 'models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
filenames = ['valid_input_ids.json', 'valid_lm_labels.json', 'valid_mc_labels.json', 'valid_mc_token_ids.json']

url = "https://persona-dataset.s3.amazonaws.com/{}"

data = []

for name in filenames:
    full_url = url.format(name)
    json_data = requests.get(full_url).json()
    data.append(np.array(json_data))
    print("Done")

input_ids, lm_labels, mc_labels, mc_token_ids = data

print(lm_labels.shape)
print(input_ids.shape)

print(mc_token_ids.shape)
print(mc_labels.shape)

# index = 10
# print(index) 
f1s = np.array([])
perplexitys = np.array([])
top_1s = np.array([])
# input_ids = input_ids[:index]
# lm_labels = lm_labels[:index]
# mc_labels = mc_labels[:index]
# mc_token_ids = mc_token_ids[:index]

if not os.path.isdir(model_folder):
    gpt2.download_gpt2(model_name = '117M')

strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
batch_size = None
with strategy.scope():
    time1 = time.time()
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, batch_size=batch_size)
    i = 0
    while i < 40:
        #print("Done")
        lm_logits, mc_logits = model.predict([input_ids[i:i+4], mc_token_ids[i:i+4]], batch_size=4)
        # has_started = True
        #lm_logits, mc_logits = model.predict([input_ids, mc_token_ids], batch_size=batch_size*4)
        lm_logits = tf.convert_to_tensor(lm_logits)
        mc_logits = tf.convert_to_tensor(mc_logits)
        lm_labels = tf.convert_to_tensor(lm_labels[i:i+4])
        mc_labels = tf.convert_to_tensor(mc_labels[i:i+4])

        ppl = K.eval(perplexity_lm(lm_labels, lm_logits))
        f1 = K.eval(f1_score_lm(lm_labels, lm_logits))
        #top_1 = K.eval(top_1_lm(lm_labels, lm_logits))
        top_1_mc = K.eval(top_1_mc(mc_labels, mc_logits))

        perplexitys = np.concatenate([perplexitys, ppl], axis=0) if i else ppl
        f1s = np.concatenate([f1s, f1], axis=0) if i else f1
        top_1s = np.concatenate([top_1s, top_1_mc], axis=0) if i else top_1_mc
        i += 4

    # print("Perplexity", ppl)
    # print("F1 Score", f1)
    # print("Hits@1", top_1_mc)

    metrics = {
        'ppl': perplexitys.tolist(),
        'f1': f1s.tolist(),
        'top': top_1s.tolist()
    }

    import json

    with open("metrics.json", 'w') as f:
        json.dump(metrics, f)

    print(time.time() - time1)







# print(current_lm.shape)
# print(current_mc.shape)

# lm_logits = tf.convert_to_tensor(current_lm)
# mc_logits = tf.convert_to_tensor(current_mc)


# lm_logits = tf.convert_to_tensor(lm_logits)
# mc_logits = tf.convert_to_tensor(mc_logits)
# lm_labels = tf.convert_to_tensor(lm_labels)
# mc_labels = tf.convert_to_tensor(mc_labels)

# ppl = perplexity_lm(lm_labels, lm_logits)
# f1 = f1_score_lm(lm_labels, lm_logits)
# top_1 = top_1_mc(mc_labels, mc_logits)

# print("Perplexity", K.eval(ppl))
# print("F1 Score", K.eval(f1))
# print("Hits@1", K.eval(top_1))

# strategy = tf.distribute.MirroredStrategy()

# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# batch_size = 2
# with strategy.scope():
    
#     print("starting fit")
#     history_output = model.fit(
#         {
#             'LMInput': input_ids,
#             'MCInput': mc_token_ids
#         },
#         {
#             'LMOutput': lm_labels,
#             'MCOutput': mc_labels
#         },
#         batch_size=batch_size * strategy.num_replicas_in_sync,
#         epochs=2,
#         callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
#                                        save_weights_only=True)]
#     )
#     # import json

#     # with open('training_history.json', 'w') as f:
#     #     json.dump(history_output.history, f)
