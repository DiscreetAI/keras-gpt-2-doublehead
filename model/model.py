from gpt2doubleheadsmodel import GPT2DoubleHeadsModel
from gpt2config import GPT2Config
import tensorflow as tf
import os

model_folder = '../models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')

config = {
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "vocab_size": 50257,
  "resid_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "attn_pdrop": 0.1,
  "output_hidden_states": True,
  "output_attentions": True
}
seq_len = None
n_ctx = config['n_ctx']
n_embd = config['n_embd']

config = GPT2Config(config)

model = GPT2DoubleHeadsModel(config)

gpt2model = model.transformer

gpt2model.wte.set_weights([
    tf.train.load_variable(checkpoint_path, 'model/wte:0'),
])
gpt2model.wpe.set_weights([
    tf.train.load_variable(checkpoint_path, 'model/wpe:0')[:seq_len, :],
])
block_list = gpt2model.block_list
for i, block in enumerate(block_list):
    block.ln_1.set_weights([
        tf.train.load_variable(checkpoint_path, 'model/h%d/ln_1/g:0' % i),
        tf.train.load_variable(checkpoint_path, 'model/h%d/ln_1/b:0' % i),
    ])
    block.attn.c_attn.set_weights([
        tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_attn/w:0' % i)[0],
        tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_attn/b:0' % i)
    ])
    block.attn.c_proj.set_weights([
        tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_proj/w:0' % i)[0],
        tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_proj/b:0' % i),
    ])
    block.ln_2.set_weights([
        tf.train.load_variable(checkpoint_path, 'model/h%d/ln_2/g:0' % i),
        tf.train.load_variable(checkpoint_path, 'model/h%d/ln_2/b:0' % i),
    ])
    block.mlp.c_fc.set_weights([
        tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_fc/w:0' % i)[0],
        tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_fc/b:0' % i)
    ])
    block.mlp.c_proj.set_weights([
        tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_proj/w:0' % i)[0],
        tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_proj/b:0' % i)
    ])
gpt2model.ln_f.get_layer(name='Norm').set_weights([
    tf.train.load_variable(checkpoint_path, 'model/ln_f/g:0'),
    tf.train.load_variable(checkpoint_path, 'model/ln_f/b:0'),
])

print(model)