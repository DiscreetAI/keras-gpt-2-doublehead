# coding=utf8
from __future__ import unicode_literals
import json
import codecs
import regex as re

import os
import six
from io import open


__all__ = ['BytePairEncoding', 'get_bpe_from_files']


try:
    chr = unichr
except Exception as e:
    '''No need to use `unichr` in Python 3'''


class BytePairEncoding(object):

    def __init__(self,
                 token_dict,
                 bpe_rank):
        """Encode and decode of BPE.

        :param token_dict: Maps from encoded token to indices.
        :param bpe_rank: Maps from byte pair to an integer rank.
        """
        self.token_dict = token_dict
        self.token_dict_inv = {v: k for k, v in self.token_dict.items()}
        self.bpe_rank = bpe_rank
        self.byte_encoder = self.init_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.token_pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        self.cache = {}
        self.unk_token = None

    @staticmethod
    def init_byte_encoder():
        codes = list(range(ord("!"), ord("~") + 1)) +\
                list(range(ord("¡"), ord("¬") + 1)) +\
                list(range(ord("®"), ord("ÿ") + 1))
        byte_encoder = {code: chr(code) for code in codes}
        shift = 0
        for code in range(2 ** 8):
            if code not in byte_encoder:
                byte_encoder[code] = chr(2 ** 8 + shift)
                shift += 1
        return byte_encoder

    def get_bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        chars = list(token)
        while len(chars) > 0:
            min_pair, min_rank = None, float('inf')
            for i in range(1, len(chars)):
                pair = (chars[i - 1], chars[i])
                rank = self.bpe_rank.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None or min_pair not in self.bpe_rank:
                break
            last, tail = chars[0], 1
            for index in range(1, len(chars)):
                if (last, chars[index]) == min_pair:
                    chars[tail - 1] = last + chars[index]
                    last = last + chars[index]
                else:
                    chars[tail - 1] = last
                    tail += 1
                    last = chars[index]
            chars[tail - 1] = last
            chars = chars[:tail]
        self.cache[token] = chars
        return chars

    def encode(self, text):
        indices = []
        for token in re.findall(self.token_pattern, text):
            token = bytearray(token.encode('utf-8'))
            chars = ''.join(self.byte_encoder[code] for code in token)
            indices += [self.token_dict[token] for token in self.get_bpe(chars)]
        return indices

    def decode(self, tokens):
        text = ''.join([self.token_dict_inv[token] for token in tokens])
        return bytearray([self.byte_decoder[byte] for byte in text]).decode('utf-8', errors='replace')


    def add_tokens(self, new_tokens):
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.
        Args:
            new_tokens: list of string. Each string is a token to add. Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).
        Returns:
            Number of tokens added to the vocabulary.
        Examples::
            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        if not new_tokens:
            return 0

        to_add_tokens = []
        for token in new_tokens:
            assert isinstance(token, str) or (six.PY2 and isinstance(token, unicode))
            if token != self.unk_token and \
                    self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token):
                to_add_tokens.append(token)
                logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
        added_tok_decoder = {v:k for k, v in added_tok_encoder.items()}
        self.token_dict.update(added_tok_encoder)
        self.token_dict_inv.update(added_tok_decoder)

        return len(to_add_tokens)

    def add_special_tokens(self, special_tokens_dict):
        """
        Add a dictionary of special tokens (eos, pad, cls...) to the encoder and link them
        to class attributes. If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).
        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).
        Returns:
            Number of tokens added to the vocabulary.
        Examples::
            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')
            special_tokens_dict = {'cls_token': '<CLS>'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
            assert tokenizer.cls_token == '<CLS>'
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES
            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all(isinstance(t, str) or (six.PY2 and isinstance(t, unicode)) for t in value)
                added_tokens += self.add_tokens(value)
            else:
                assert isinstance(value, str) or (six.PY2 and isinstance(value, unicode))
                added_tokens += self.add_tokens([value])
            logger.info("Assigning %s to the %s key of the tokenizer", value, key)
            setattr(self, key, value)

        return added_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if isinstance(tokens, str) or (six.PY2 and isinstance(tokens, unicode)):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        if len(ids) > self.max_len:
            logger.warning("Token indices sequence length is longer than the specified maximum sequence length "
                           "for this model ({} > {}). Running this sequence through the model will result in "
                           "indexing errors".format(len(ids), self.max_len))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token in self.token_dict:
            return self.token_dict[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError


def get_bpe_from_files(encoder_path, vocab_path):
    """Get initialized BPE.

    :param encoder_path: Path to 'encoder.json'.
    :param vocab_path: Path to 'vocab.bpe'
    :return: The object from encode and decode strings.
    """
    with codecs.open(encoder_path, 'r', 'utf8') as reader:
        token_dict = json.load(reader)
    bpe_rank = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        reader.readline()
        for rank, line in enumerate(reader):
            line = line.strip()
            if line:
                bpe_rank[tuple(line.split())] = rank
    return BytePairEncoding(token_dict, bpe_rank)
