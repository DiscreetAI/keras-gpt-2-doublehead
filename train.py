import os
import sys
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
from keras.models import load_model
from keras.callbacks import BaseLogger, History
import tensorflow as tf
import numpy as np
from collections import defaultdict
import urllib

model_folder = 'models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')

import gpt_2_simple as gpt2

if not os.path.isdir(model_folder):
    gpt2.download_gpt2(model_name = '117M')



print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)
print('Generate text...')
# output = generate(model, bpe, ['From the day forth, my arm'], length=20, top_k=40)
from itertools import chain
import re
import numpy as np
import json

# Let's define our contexts and special tokens
persona = [["i like playing football."],
           ["i am from NYC."]]
history = [["hello how are you?"],
           ["i am fine thanks."]]
reply = ["great to hear"]
persona = [re.findall(r"[\w']+|[.,!?;]", sentence[0]) for sentence in persona]
history = [re.findall(r"[\w']+|[.,!?;]", sentence[0]) for sentence in history]
reply = re.findall(r"[\w']+|[.,!?;]", reply[0])
speaker1, speaker2 = "You said: ", "I said: "


MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    speaker1, speaker2 = tokenizer.convert_tokens_to_ids('You said: '), tokenizer.convert_tokens_to_ids('I said: ')

    instance = {}
    # persona = [person.split() for person in persona]
    # history = [hist.split() for hist in history]
    # reply = reply.split()
    sequence = [list(chain(*persona))] + history + [reply]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence

def get_data_loaders(personachat, tokenizer, args_num_candidates=1, args_personality_permutations=1, args_max_history=2):
    """ Prepare the dataset for training and evaluation """
   

    print("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args_num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args_num_candidates, num_candidates)
        if dataset_name == 'valid':
            continue
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args_personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args_max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance, _ = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": []}
    for dataset_name, dataset in datasets.items():
        if dataset_name == 'valid':
            continue
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids('<pad>'))
        for input_name in MODEL_INPUTS:
            tensor = np.array(dataset[input_name])
            # if input_name == "mc_ldsfaabels":
            #     tensor = tensor.reshape((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            dataset[input_name] = tensor

    return datasets

import json
from pytorch_pretrained_bert import cached_path
from pytorch_pretrained_bert import GPT2Tokenizer
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
url = "s3://datasets.huggingface.co/personachat/personachat_self_original.json"

# Download and load JSON dataset
# personachat_file = "dataset.json"
# with open(personachat_file, "r", encoding="utf-8") as f:
#     dataset = json.loads(f.read())



# # dataset = {'train': dataset['train']}
# # dataset['train'] = dataset['train'][:1]
# # print('\n')
# # print(dataset[0]['utterances'][1])
# # print('\n')
# # print(dataset[0]['utterances'][2])
# # Tokenize and encode the dataset using our loaded GPT tokenizer
# def tokenize(obj):
#     if isinstance(obj, str):
#         return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
#     if isinstance(obj, dict):
#         return dict((n, tokenize(o)) for n, o in obj.items())
#     return list(tokenize(o) for o in obj)

# # print("Tokenizing dataset...") 
# # dataset = tokenize(dataset)

# # with open('dataset.json', "w", encoding="utf-8") as f:
# #     f.write(json.dumps(dataset))
# print(len(dataset['train']))
# # with open('dataset.json', 'r') as f:
# #     personachat = json.loads(f.read())
# datasets = get_data_loaders(dataset, tokenizer)

# arr = datasets['train']
# input_ids = arr['input_ids']
# mc_token_ids = arr['mc_token_ids']
# lm_labels = arr['lm_labels']
# mc_labels = arr['mc_labels']

# np.save('input_ids.npy', input_ids)
# np.save('mc_token_ids.npy', mc_token_ids)
# np.save('lm_labels.npy', lm_labels)
# np.save('mc_labels.npy', mc_labels)

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

# import urllib

# import requests

# filenames = ['input_ids.npy', 'lm_labels.npy', 'mc_labels.npy', 'mc_token_ids.npy']

# import os
# import boto3
# import botocore

# files = filenames

# bucket = 'persona-dataset'

# s3 = boto3.resource('s3')

# for file in files:
#    try:
#        s3.Bucket(bucket).download_file(file, os.path.basename(file))
#    except botocore.exceptions.ClientError as e:
#        if e.response['Error']['Code'] == "404":
#            print("The object does not exist.")
#        else:
#            raise

# print('Load model from checkpoint...')
# model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
# history_output = model.fit(
#     {
#         'LMInput': input_ids,
#         'MCInput': mc_token_ids
#     },
#     {
#         'LMOutput': lm_labels,
#         'MCOutput': mc_labels
#     },
#     batch_size=1,
#     epochs=3,
#     callbacks=[BaseLogger()]
# )

# import json

# with open('training_history.json', 'w') as f:
#     json.dump(history_output.history, f)

# model.save('trained_model.h5')