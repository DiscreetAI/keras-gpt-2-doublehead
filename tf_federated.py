import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
import tensorflow_federated as tff
import numpy as np
import os
import sys
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
from keras.models import load_model
from keras.callbacks import BaseLogger, History
from collections import defaultdict
import urllib
import gpt_2_simple as gpt2

input_ids = np.load('input_ids.npy', allow_pickle=True)
mc_token_ids = np.load('mc_token_ids.npy', allow_pickle=True)
lm_labels = np.load('lm_labels.npy', allow_pickle=True)
mc_labels = np.load('mc_labels.npy', allow_pickle=True)


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

datasets = [tuple(dataset) for dataset in datasets]
datasets = [tf.data.Dataset.from_tensor_slices(dataset) for dataset in datasets]

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
    return tff.learning_from_compiled_keras_model(model, sample_batch)

# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(model_fn)
state = trainer.initialize()
for _ in range(5):
  state, metrics = trainer.next(state, train_data)
  print (metrics.loss)