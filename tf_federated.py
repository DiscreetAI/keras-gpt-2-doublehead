import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
import tensorflow_federated as tff
import tensorflow.python.keras.backend as K
import numpy as np
import os
import sys
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import BaseLogger, History
from collections import defaultdict
import urllib
import gpt_2_simple as gpt2
import requests
from collections import OrderedDict

filenames = ['input_ids.json', 'lm_labels.json', 'mc_labels.json', 'mc_token_ids.json']

url = "https://persona-dataset.s3.amazonaws.com/{}"

data = []

for name in filenames:
    full_url = url.format(name)
    json_data = requests.get(full_url).json()
    data.append(np.array(json_data))
    print("Done")

input_ids, lm_labels, mc_labels, mc_token_ids = data


# with open('preprocessed_dataset.json', 'w') as f:
#     personachat = json.dump(datasets, f)

print(lm_labels.shape)
print(input_ids.shape)

print(mc_token_ids.shape)
print(mc_labels.shape)

num_clients = 5

batches = [np.array_split(input_ids, num_clients), np.array_split(lm_labels, num_clients), np.array_split(mc_token_ids, num_clients), np.array_split(mc_labels, num_clients)]

assert len(batches) == 4
assert len(batches[0]) == num_clients

datasets = list(zip(*batches))

assert len(datasets) == num_clients
assert len(datasets[0]) == 4

a, b, c, d = datasets[0]
print(a.shape, b.shape, c.shape, d.shape)
tf_datasets = []
# for input_ids, lm_labels, mc_token_ids, mc_labels in datasets:
#     tf_datasets.append()

def dataset_map(input_ids, lm_labels, mc_token_ids, mc_labels):
    result = OrderedDict([
        ('x',  OrderedDict([
            ('LMInput', K.expand_dims(input_ids)),
            ('MCInput', K.expand_dims(mc_token_ids))
        ])),
        ('y', K.concatenate([K.expand_dims(lm_labels), K.expand_dims(mc_labels)], axis=0))
    ])
    print(result.get('y')[0])
    return result

#datasets = [tuple(dataset) for dataset in datasets]
datasets = [tf.data.Dataset.from_tensor_slices(dataset) for dataset in datasets]
datasets = [dataset.map(dataset_map) for dataset in datasets]

train_data = datasets

# Grab a single batch of data so that TFF knows what data looks like.
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(train_data[0]).next())

def model_fn():
    model_folder = 'models/117M'
    config_path = os.path.join(model_folder, 'hparams.json')
    checkpoint_path = os.path.join(model_folder, 'model.ckpt')
    encoder_path = os.path.join(model_folder, 'encoder.json')
    vocab_path = os.path.join(model_folder, 'vocab.bpe')

    if not os.path.isdir(model_folder):
        gpt2.download_gpt2(model_name = '117M')

    print('Load BPE from files...')
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    return tff.learning.from_compiled_keras_model(model, sample_batch)

# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(model_fn)
state = trainer.initialize()
for _ in range(5):
  state, metrics = trainer.next(state, train_data)
  print (metrics.loss)