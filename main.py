import json
import torch
import matplotlib.pyplot as plt

"""
0 = floor tom
1 = snare
2 = closed hihat
3 = ride cymbal
4 = crash cymbal
5 = splash cymbal
6 = high-pitched tom
7 = medium-pitched tom
8 = low-pitched tom
9 = floor tom and closed hihat
10 = snare and closed hihat
11 = open hihat
12 = floor tom and ride
13 = snare and ride
14 = silence
15 = close hihat (hihat foot)
16 = floor tom and open hihat
17 = snare and open hihat
18 = <S>
19 = <E>
"""
# open json file
with open("tabs.json") as f:
    songs = json.load(f)

# initialize our ocurrencies matrix (where we store each tuple of consequent characters)
N = torch.zeros((20, 20), dtype=torch.int32)
chars = [num for num in range(20)]

for song in songs:
    for ch1, ch2 in zip(song, song[1:]):
        N[ch1, ch2] += 1


plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(20):
    for j in range(20):
        plt.text(j, i, (str(j) + "," + str(i)), ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")
plt.show()

"""
p = N[0].float()
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()

torch.multinomial(p, num_samples=100, replacement=True, generator=g)

P = (N + 1).float()
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
    print("".join(out))
"""
