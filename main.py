# inference code, but in .py format

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import LanguageModel

user_input = input("How many characters do you want to generate? (default 250): ")
max_new_tokens = int(user_input) if user_input else 250

with open("data/input.txt", "r") as text:
    text = text.read()

# get unique chars in training data
chars = sorted(list(set(text)))
vocab_size = len(chars)

# map string to integers (idx)
stoi = { ch:i for i,ch in enumerate(chars) } # string to int
itos = { i:ch for i,ch in enumerate(chars) } # int to string

# char encode/decode
encode = lambda s: [stoi[c] for c in s] # string -> list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # list of ints -> string

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LanguageModel()
model.to(device)

if (device == 'cuda'):
    model.load_state_dict(torch.load('checkpoints/model.pth'))
else:
    model.load_state_dict(torch.load('checkpoints/model.pth', map_location=torch.device('cpu')))

# generate text
starting_prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
result, probs_list = model.generate(starting_prompt, max_new_tokens=max_new_tokens)
result = decode(result[0].tolist())
print(result)


user_input = input("Would you like to see how the generated words compare to the highest probability words over time? (y/n): ")

if (user_input.lower() == 'y'):
    actual_probs = []
    max_probs = []

    for new_token_idx in range(max_new_tokens):
        # Convert tensor to numpy array
        probs = probs_list[new_token_idx][0].cpu().detach().numpy() # 0th batch

        # Get the actual word chosen and its probability
        actual_word = result[new_token_idx] # get the word at the current index
        actual_prob = probs[stoi[actual_word]]
        actual_probs.append(actual_prob)

        # Get the maximum probability
        max_prob = np.max(probs)
        max_probs.append(max_prob)

    # Create a time axis
    time = np.arange(max_new_tokens)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(time, actual_probs, label='Actual Word', marker='o')
    plt.plot(time, max_probs, label='Highest Probability', marker='o')
    plt.xlabel('Step')
    plt.ylabel('Probability')
    plt.title('Probability of Actual Word vs. Highest Probability Over Time')
    plt.legend()
    plt.show()