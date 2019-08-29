import os
import sys
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
import tensorflow as tf
import numpy as np

model_folder = 'models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


print('Load model from checkpoint...')
#model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)
print('Generate text...')
# output = generate(model, bpe, ['From the day forth, my arm'], length=20, top_k=40)
from itertools import chain
import re


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

def build_inputs(persona, history, reply):
    # Build our sequence by adding delimiters and concatenating
    sequence = [list(chain(*persona))] + history + [reply]
    sequence = [sequence[0]] + [ [speaker2 if (len(sequence)-i) % 2 else speaker1] + s
                                for i, s in enumerate(sequence[1:])]
    # Build our word, segments and position inputs from the sequence
    words = list(chain(*sequence))                          # word tokens
    segments = [speaker2 if i % 2 else speaker1             # segment tokens
                for i, s in enumerate(sequence) for _ in s]
    position = list(range(len(words)))                      # position tokens
    return words, segments, position, sequence

words, segments, position, sequence = build_inputs(persona, history, reply)

# >>> print(sequence)  # Our inputs looks like this:
# [['<bos>', 'i', 'like', 'playing', 'football', '.', 'i', 'am', 'from', 'NYC', '.'],
#  ['<speaker1>', 'hello', 'how', 'are', 'you', '?'],
#  ['<speaker2>', 'i', 'am', 'fine', 'thanks', '.'],
#  ['<speaker1>', 'great', 'to', 'hear', '<eos>']]

# Tokenize words and segments embeddings:
words = [bpe.encode(word) for word in words]
segments = [bpe.encode(segment) for segment in segments]

# If you are using the 117M model and top_k equals to 1, then the result would be:
# "From the day forth, my arm was broken, and I was in a state of pain. I was in a state of pain,"


# Let's add a distractor to our previously defined persona, history and reply
distractor = ["sorry", "to", "hear", "that"]

# Build & tokenize inputs ending with our distractor like we did with the gold reply
words_distractor, segments_distractor, _, _ = build_inputs(persona, history, distractor)
words_distractor = [bpe.encode(word) for word in words_distractor]
segments_distractor = [bpe.encode(segment) for segment in segments_distractor]

# Prepare our language modeling targets: keep only the reply segment, -1 on the rest
lm_targets = ([-1] * sum(len(s) for s in sequence[:-1])) \
             + [-1] + [bpe.encode(word) for word in sequence[-1][1:]]
lm_distractor = [-1] * len(words_distractor)

# Store the position of the last tokens for the next-sentence prediction loss
last_token = len(words) - 1
last_token_distractor = len(words_distractor) - 1

# Now we can pad reply and distractor inputs and targets to the same length
padding_length = max(len(words), len(words_distractor))
def pad(x, padding):
    return x + [padding] * (padding_length - len(x))

(words, words_distractor,
 segments, segments_distractor) = [pad(x, bpe.encode('<pad>'))
                                   for x in (words, words_distractor,
                                             segments, segments_distractor)]

(lm_targets, lm_distractor) = [pad(x, -1) for x in (lm_targets, lm_distractor)]
 
# And gather reply and distractor inputs to build the input tensors:
# words tokens
input_ids = np.array([[words, words_distractor]])
# segment tokens
token_type_ids = np.array([[segments, segments_distractor]])
# Positions tokens can be automatically created by the model as (0, 1, ..., N)
# Last tokens location
mc_token_ids = np.array([[last_token, last_token_distractor]])
# Language modeling labels
lm_labels = np.array([[lm_targets, lm_distractor]])
# Next-sentence prediction labels
mc_labels = np.array([0])  # Gold reply is 1st (index 0)