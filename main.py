import json
import torch
import matplotlib.pyplot as plt


with open('tabs.json') as f:
    songs = json.load(f)

b = {}
for song in songs:
    for ch1, ch2 in zip(song, song[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

N = torch.zeros((8, 8), dtype=torch.int32)
chars = '01234567'
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

for w in songs:
    for ch1, ch2 in zip(song, song[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(8):
    for j in range(8):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

p = N[0].float()
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()

torch.multinomial(p, num_samples=100, replacement=True, generator=g)

P = (N+1).float()
P /= P.sum(1, keepdims=True)


g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
