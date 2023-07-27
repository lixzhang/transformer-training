from transformers import GPT2Config, GPT2LMHeadModel
import torch
import time
from datetime import timedelta

torch.manual_seed(1337)

# parameters
max_iters = 5000
eval_interval = 500
batch_size = 64
block_size = 256
max_new_tokens = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 3

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

data = torch.tensor(encode(text), dtype=torch.long)
# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # modified version of Karpathy
    # About labels in GPT2LMHeadModel() being shifted internally
	# https://github.com/alexcpn/tranformer_learn/blob/gpt-loss-learn/gpt2_train_model.py
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i:i+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            outputs = model(input_ids=X, labels=Y)
            loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

def generate(idx, max_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        idx_next = model.generate(idx_cond, max_new_tokens=1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next[:, -1:]), dim=1) # (B, T+1)
    return idx

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
model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

start_time = time.monotonic()
for iter in range(max_iters): # increase number of steps for good results... 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    outputs = model(input_ids=xb, labels=yb)
    loss = outputs.loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
with open('training_time.txt', 'w', encoding='utf-8') as f:
	f.write(str(timedelta(seconds=end_time - start_time)))

save_direrctory = 'saved_model'
model.save_pretrained(save_direrctory)
model = GPT2LMHeadModel.from_pretrained(save_direrctory)
context = torch.zeros((1, 1), dtype=torch.long, device='cpu')
generated = generate(context, max_new_tokens)[0]
generated_text = decode(generated.tolist())
with open('generated.txt', 'w', encoding='utf-8') as f:
	f.write(generated_text)
