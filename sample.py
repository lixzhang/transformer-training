from transformers import GPT2Config, GPT2LMHeadModel
import torch
import time
from datetime import timedelta
import os

def generate(idx, max_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        idx_next = model.generate(idx_cond, max_new_tokens=1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next[:, -1:]), dim=1) # (B, T+1)
    return idx


out_dir = 'out_shakespears_char'
save_direrctory = 'saved_model'

# parameters
max_iters = 5000
eval_interval = 500
batch_size = 64
block_size = 256
max_new_tokens = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6

# get data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Define the configuration
config = GPT2Config(
    vocab_size=vocab_size+1,  # your desired vocab size
    # other parameters
    bos_token_id=vocab_size,
    pad_token_id=vocab_size,
    eos_token_id=vocab_size,
    n_positions=block_size,
    # n_ctx=1024,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    # resid_pdrop=0.1,
    # embd_pdrop=0.1,
    # attn_pdrop=0.1,
    # layer_norm_epsilon=1e-5,
    # initializer_range=0.02,
)

# Create a new model with this configuration
model = GPT2LMHeadModel(config)
model = GPT2LMHeadModel.from_pretrained(os.path.join(out_dir, save_direrctory))
model.to(device)

n_samples = 10

for _ in range(n_samples):
    context = torch.zeros((1, 1), dtype=torch.long)
    context = context.to(device)
    generated = generate(context, max_new_tokens)[0]
    generated_text = decode(generated.tolist())
    with open(os.path.join(out_dir, 'generated.txt'), 'a', encoding='utf-8') as f:
        f.write(generated_text)
        f.write('\n--------------------------\n')
