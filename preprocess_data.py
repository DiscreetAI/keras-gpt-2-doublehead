import os
import sys
import numpy as np
from collections import defaultdict
import urllib

from itertools import chain
import re
import numpy as np
import json



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
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids('<pad>'))
        for input_name in MODEL_INPUTS:
            tensor = dataset[input_name]
            dataset[input_name] = np.array(tensor)

    return datasets

import json
from pytorch_pretrained_bert import cached_path
from pytorch_pretrained_bert import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
url = "s3://datasets.huggingface.co/personachat/personachat_self_original.json"

personachat_file = cached_path(url)
with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())

# Tokenize and encode the dataset using our loaded GPT tokenizer
def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)

print("Tokenizing dataset..") 
dataset = tokenize(dataset)

with open('dataset.json', "w", encoding="utf-8") as f:
    f.write(json.dumps(dataset))
datasets = get_data_loaders(dataset, tokenizer)

arr = datasets['valid']
input_ids = arr['input_ids'].tolist()
mc_token_ids = arr['mc_token_ids'].tolist()
lm_labels = arr['lm_labels'].tolist()
mc_labels = arr['mc_labels'].tolist()


with open('input_ids.json', 'w') as f:
    json.dump(input_ids, f)

with open('lm_labels.json', 'w') as f:
    json.dump(lm_labels, f)

with open('mc_token_ids.json', 'w') as f:
    json.dump(mc_token_ids, f)

with open('mc_labels.json', 'w') as f:
    json.dump(mc_labels, f)