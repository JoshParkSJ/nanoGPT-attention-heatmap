# Nano-GPT with Simple Interpretability



## Transformer Architecture

<img src="assets/transformer.png" alt="Transformer architecture" width="500" height="750">




## Implementation Walkthrough


#### Training data
Use text file containing shakespeare poems
```
! wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

Use a 90/10 train/val split
```
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

```

Data loading method to get batch of training text and label. Shape of x and y are both (batch_size, context_length) where context length is an arbitrary number we choose. The model will be trained to predict up to context_length (1 char, 2 char, ..., context_length char)
```
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

#### Vocabulary

Create maps for string to int `stoi` and int to string `itos` by getting unique chars in training data

```
chars = sorted(list(set(text)))
vocab_size = len(chars)

```

Then map chars to integers (idx)
```
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 
```

Encode: string -> list of ints

Decode: list of ints -> string

#### Multi-head Attention (!!!)



```
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length))) # add a persistent buffer (self.tril) to the nn.module which will not be updated during backpropagation

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
```

```
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # multiple heads in parallel
        self.proj = nn.Linear(n_embd, n_embd) # residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # run heads in parallel and concatenate them together in the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) # output of self-attention
        out = self.dropout(self.proj(out)) # apply projection (projection back into residual pathway). Projection is just a linear transformation of the output of self-attention
        return out
```