import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import BaseLogger, History
import tensorflow as tf
import numpy as np
from collections import defaultdict
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
import requests
import gpt_2_simple as gpt2
from keras_gpt_2 import perplexity_lm, f1_score_lm, top_1_mc, top_1_lm
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
import time
import simplejson as json
import sys


if len(sys.argv) != 2:
    print('Enter model number!')
    sys.exit(-1)

epoch_number = int(sys.argv[1])

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 

model_folder = 'models/117M'
config_path = os.path.join(model_folder, 'hparams.json')

encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')
checkpoint_dir = './training_checkpoints'
if epoch_number == 0:
    checkpoint_path = os.path.join(model_folder, 'model.ckpt')
    already_trained = False
else:
    already_trained=True
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{epoch_number}")
print(epoch_number)
print(checkpoint_path)
filenames = ['valid_input_ids.json', 'valid_lm_labels.json', 'valid_mc_labels.json', 'valid_mc_token_ids.json']

url = "https://persona-dataset.s3.amazonaws.com/{}"

data = []

batch_size = 128
num_points_to_eval = (156020 // 128) * 128
for name in filenames:
    full_url = url.format(name)
    json_data = requests.get(full_url).json()
    data.append(np.array(json_data))
    print("Done")

input_ids, lm_labels, mc_labels, mc_token_ids = data

#print(lm_labels.shape)
#print(input_ids.shape)

#print(mc_token_ids.shape)
#print(mc_labels.shape)

class Timer():
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t

index = num_points_to_eval
# print(index) 
f1s = []
perplexitys = []
top_1s = []
input_ids = input_ids[:index]
lm_labels = lm_labels[:index]
mc_labels = mc_labels[:index]
mc_token_ids = mc_token_ids[:index]

if not os.path.isdir(model_folder):
    gpt2.download_gpt2(model_name = '117M')

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, batch_size=None, already_trained=already_trained)
print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)
print('Generate text...')
output = generate(model, bpe, ['From the day forth, my arm'], length=20, top_k=40)

# if epoch_number == 0:
#     strategy = tf.distribute.MirroredStrategy()
# elif epoch_number == 1:
#     strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3'])
# else:
#     strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:4', '/device:GPU:5', '/device:GPU:6', '/device:GPU:7'])

# #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# with strategy.scope():
    
#     timer = Timer()
#     i = 0

#     print(f"Time since last iteration to do: {timer()}")
#     #print("Done")
#     # print(input_ids[i:i+batch_size].shape)
#     # print(mc_token_ids[i:i+batch_size].shape)
#     # print(lm_labels[i:i+batch_size].shape)
#     # print(mc_labels[i:i+batch_size].shape)
#     metrics = model.evaluate(
#         x = {
#             'LMInput': input_ids,
#             'MCInput': mc_token_ids
#         }, 
#         y = {
#             'LMOutput': lm_labels,
#             'MCOutput': mc_labels
#         }, 
#         batch_size=batch_size
#     ) 
#     print(model.metrics_names)
#     print(metrics)
        

#     # print("Perplexity", ppl)
#     # print("F1 Score", f1)
#     # print("Hits@1", top_1_mc)

#     print(f"Total time: {timer.total_time}")
#     metrics = {
#         'loss': metrics[0],
#         'ppl': metrics[3],
#         'f1': metrics[5],
#         'top': metrics[4]
#     }
    
#     #print(time.time() - time1, "total time")
#     print(metrics)

#     with open("metrics.json", 'w') as f:
#         json.dump(metrics, f, use_decimal=True)

    







# # print(current_lm.shape)
# # print(current_mc.shape)

# # lm_logits = tf.convert_to_tensor(current_lm)
# # mc_logits = tf.convert_to_tensor(current_mc)


# # lm_logits = tf.convert_to_tensor(lm_logits)
# # mc_logits = tf.convert_to_tensor(mc_logits)
# # lm_labels = tf.convert_to_tensor(lm_labels)
# # mc_labels = tf.convert_to_tensor(mc_labels)

# # ppl = perplexity_lm(lm_labels, lm_logits)
# # f1 = f1_score_lm(lm_labels, lm_logits)
# # top_1 = top_1_mc(mc_labels, mc_logits)

# # print("Perplexity", K.eval(ppl))
# # print("F1 Score", K.eval(f1))
# # print("Hits@1", K.eval(top_1))

# # strategy = tf.distribute.MirroredStrategy()

# # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# # batch_size = 2
# # with strategy.scope():
    
# #     print("starting fit")
# #     history_output = model.fit(
# #         {
# #             'LMInput': input_ids,
# #             'MCInput': mc_token_ids
# #         },
# #         {
# #             'LMOutput': lm_labels,
# #             'MCOutput': mc_labels
# #         },
# #         batch_size=batch_size * strategy.num_replicas_in_sync,
# #         epochs=2,
# #         callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
# #                                        save_weights_only=True)]
# #     )
# #     # import json

# #     # with open('training_history.json', 'w') as f:
# #     #     json.dump(history_output.history, f)
