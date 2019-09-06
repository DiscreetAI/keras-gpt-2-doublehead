import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import BaseLogger, History
import tensorflow as tf
import numpy as np
from collections import defaultdict
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
import requests
import gpt_2_simple as gpt2
from keras_gpt_2 import Metrics

from tensorflow.python.client import device_lib
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
filenames = ['input_ids.json', 'lm_labels.json', 'mc_labels.json', 'mc_token_ids.json']

url = "https://persona-dataset.s3.amazonaws.com/{}"

data = []

for name in filenames:
    full_url = url.format(name)
    json_data = requests.get(full_url).json()
    data.append(np.array(json_data))
    print("Done")

input_ids, lm_labels, mc_labels, mc_token_ids = data

index = (131438 // 16) * 16
print(index) 

input_ids = input_ids[:index]
lm_labels = lm_labels[:index]
mc_labels = mc_labels[:index]
mc_token_ids = mc_token_ids[:index]

if not os.path.isdir(model_folder):
    gpt2.download_gpt2(model_name = '117M')

strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
batch_size = 2
with strategy.scope():
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, batch_size=batch_size)
    print("starting fit")
    history_output = model.fit(
        {
            'LMInput': input_ids,
            'MCInput': mc_token_ids
        },
        {
            'LMOutput': lm_labels,
            'MCOutput': mc_labels
        },
        batch_size=batch_size * strategy.num_replicas_in_sync,
        epochs=2,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True)]
    )
    # import json

    # with open('training_history.json', 'w') as f:
    #     json.dump(history_output.history, f)
