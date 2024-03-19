# NanoGPT with Simple Interpretability

1. **Extract Attention Weights**: After you run your input through the model, you can extract the attention weights. These weights are part of the model's internal state, and they essentially show how much the model "paid attention" to each input token when generating each output token.

2. **Visualize Attention Weights**: You can then visualize these weights as a heatmap, where the x-axis represents input tokens, the y-axis represents output tokens, and the color of each cell represents the attention weight.

If a particular input token has high attention weights across many output tokens, that suggests it was important in generating the output.


<img src="assets/transformer.png" alt="Transformer architecture" width="500" height="750">


# Training Data
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

Data loading method to get batch of training text and label. Shape of x and y are both (batch_size, context_length) where context length is an arbitrary number we choose. The model will be trained to predict up to context_length (1 char, 2 char, ..., context_length char). 

Note that `y` is length `context_length+1` because there are `context_length` number of examples packed and the `context_length`th example needs to predict the next token (+1). This means for each `y`, the model is trained to predict the next token when the input only has 1, 2, 3, ..., `context_length` number of tokens. 

- `batch_size` sets how many independent sequences will be processed in parallel.

- `context_length` is max contenxt length for predictions

- `ix` contains randomly generated `batch_size` numbers of offsets between `0` and `len(data) - context_length`. Now we can sample from our randomized set of `ix` training data.

- `x` sample a point from `ix` and take the next `context_length` to represent one trianing data example. Stack examples in `batch_size`x`context_length` tensors 

- `y` represents target values (i.e label). Based on the same `ix` sampling, take the next token (`i+1`) which is what the model needs to predict 

```
batch_size = 4 
context_length = 8 

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,)) 
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

```

# Vocabulary

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

# High-level Overview

1. Input embedding: Each word maps to a vector

2. Positional encoding: inject positional encoding into the embedding (information about positions, i.e mimics recurrence in RNN to learn momentum)

3. Multi-head self-attention (Nx transformer blocks): can be seen as data communication followed by computation

4. Last layer norm: stabilizes the network (stable training, incremental progress) 

5. Last linear layer: output of final pointwise FNN goes through final linear layer which acts as a classifier (clasifier is as big as how many classes you have ⇒ i.e size 10k = 10k word vocab). 

6. Cross-entropy loss function: first applies a softmax function to the logits, and then computes the negative log likelihood loss. Softmax function converts logits into probabilities, ensuring that they are positive and sum to 1. The NLL then compares these probabilities with the true labels to compute the loss

The final output is a probability score for each class. On inference, the model takes the highest probable word as the "next predicted word"

```{python}
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # n_embed=32 dimensional embedding
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)] # n_layer amount of blocks
        ) 
        
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. input embedding
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd) tensor of integers

        # 2. positional encoding
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        # 3. multi-head self-attention (Nx transformer blocks)
        x = self.blocks(x) # (B,T,C)

        # 4. last layer norm
        x = self.ln_f(x) # (B,T,C)

        # 5. last linear layer
        logits = self.lm_head(x) # (B,T,vocab_size) C for channels but our case is vocab_size

        # 6. Cross-entropy loss function (softmax)
        B, T, C = logits.shape 
        logits = logits.view(B*T, C)
        targets = targets.view(B*T) # targets (B,T) tensor of integers
        loss = F.cross_entropy(logits, targets)

        return logits, loss
```

# Transformer Block

Self-attention is data communication. Feed forward is data computation (think on all the tokens independently). 

Note: although the original paper applies the layer norm after the self-attention and feed forward layers, modern implementaiton of transformers use pre-norm formulation where the layer norm is applied before the self-attention and feed forward layers.   


```
class Block(nn.Module):

    # n_embd: embedding dimension (32)
    # n_head: the number of heads we'd like
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x
```


# !!! Self-Attention !!!


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

Multi-head self-attention is 
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

