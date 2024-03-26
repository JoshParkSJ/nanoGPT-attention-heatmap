
# Attention Mechanism Implementation

## Simple weighted aggregation: Averaging
Average of all the values before curr_token
```python
for t in range(T):
    xprev = x[b,:t+1] 
    xbow[b,t] = torch.mean(xprev, 0) # mean along 0th dimension (row)
```

First batch, 8 context length, 2 letter per token = B,T,C
```python
x[0] 
tensor([[ 0.1808, -0.0700],
        [-0.3596, -0.9152],
        [ 0.6258,  0.0255],
        [ 0.9545,  0.0643],
        [ 0.3612,  1.1679],
        [-1.3499, -0.5102],
        [ 0.2360, -0.2398],
        [-0.9211,  1.5433]])
```

Last row (last token in context) of xbow is an average of all the arrays in x
```python
xbow[0]
tensor([[ 0.1808, -0.0700],
        [-0.0894, -0.4926],
        [ 0.1490, -0.3199],
        [ 0.3504, -0.2238],
        [ 0.3525,  0.0545],
        [ 0.0688, -0.0396],
        [ 0.0927, -0.0682],
        [-0.0341,  0.1332]])
```

## Simple weighted aggregation: Averaging + Triangle Matrix

Dot product of triangle and random matrix = a matrix where each row is a sum of current and previous values/tokens of random matrix. So effectively, each token is a sum of itself and previous tokens (i.e 3rd row/token of c is a sum of 3rd row/token and previous rows/tokens of b)
```python
a = torch.tril(torch.ones(3, 3))
b = torch.randint(0,10,(3,2)).float()
c = a @ b

a=
tensor([[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]])
b=
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
c=
tensor([[ 2.,  7.],
        [ 8., 11.],
        [14., 16.]])

# Remember dot product (row @ col):
c[0,0] = a[0] @ b[0] = [1,0,0] @ [2,6,6] = 2+0+0 = 2
c[0,1] = a[0] @ b[1] = [1,0,0] @ [7,4,5] = 7+0+0 = 7
c[1,0] = a[1] @ b[0] = [1,1,0] @ [2,6,6] = 2+6+0 = 8
```

We can convert the triangle matrix so that each row sums to 1:

```python
a = a / torch.sum(a, 1, keepdim=True) # sum of a in 1th dimension (col)

a=
tensor([[1.0000, 0.0000, 0.0000], # 1/1 = 1
        [0.5000, 0.5000, 0.0000], # 1/(1+1) = 1/2
        [0.3333, 0.3333, 0.3333]]) # 1/(1+1+1) = 1/3
b=
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
c=
tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]])
```

Now if we a @ b, each token is an average of itself and previous tokens. Now in the context of language model:

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow = wei @ x # (T,T) @ (B,T,C) ---> pytorch auto-creates B for (T,T) ---> (B,T,T) @ (B,T,C) ---> (B,T,C)
```

Effectively this is a sum (an aggregation) of weights 

## Weighted Aggregation: Averaging + Triangle Matrix + Softmax

When you have a triangle matrix where bottom half is 0s and top half is -inf:
```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))

wei
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```

Applying the softmax function, you get the exact same averaging effect as (triangle @ random matrix). We end up using this for self-attention because weights represent affinity (high value = high prob to be picked). So the 0 values tells the model that we will not look at those tokens ever, which makes sense here since we want the model to train the next token so the next tokens should never be looked at.
```python
wei = F.softmax(wei, dim=-1)

wei
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```

To create the actual self-attention block, we use a triangle matrix wit softmax:
```python
xbow = wei @ x
```
