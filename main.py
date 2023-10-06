import json
import torch
import matplotlib.pyplot as plt

"""
0 = starting and/or finishing token
1 = snare
2 = closed hihat
3 = ride cymbal
4 = crash cymbal
5 = splash cymbal
6 = high-pitched tom
7 = medium-pitched tom
8 = low-pitched tom
9 = kick drum and closed hihat
10 = snare and closed hihat 
11 = open hihat
12 = kick drum and ride
13 = snare and ride
14 = silence
15 = close hihat (hihat foot)
16 = kick drum and open hihat
17 = snare and open hihat
18 = kick drum
19 = kick drum and crash
"""
# open json file
with open("tabs.json") as f:
    songs = json.load(f)

# initialize our ocurrencies matrix (where we store each tuple of consequent characters)
N = torch.ones((20, 20), dtype=torch.int32)
chars = [num for num in range(20)]

for song in songs:
    for ch1, ch2 in zip(song, song[1:]):
        N[ch1, ch2] += 30

P = N.float()
P /= P.sum(1, keepdim=True)
g = torch.Generator().manual_seed(2147483647)


def show_occurrence_graph():
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for i in range(20):
        for j in range(20):
            plt.text(
                j, i, (str(j) + "," + str(i)), ha="center", va="bottom", color="gray"
            )
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")
    plt.show()


# FIRST APPROACH WITH LITTLE ACCURACY
def first_approach():
    p = N[0].float()
    p = p / p.sum()

    ix = 0
    new_tab = [0]
    for _ in range(10):
        while True:
            p = N[ix].float()
            p = p / p.sum()
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            new_tab.append(ix)
            if ix == 0:
                break
        print(new_tab)
        new_tab = []


# MAKING P MORE EFFICIENT BY CALCULATING PROBABILITY DIST. BEFOREHAND
def first_approach_efficient():
    g = torch.Generator().manual_seed(2147483647)
    ix = 0
    new_tab = [0]
    for _ in range(10):
        while True:
            p = P[ix]
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            new_tab.append(ix)
            if ix == 0:
                break
        print(new_tab)
        new_tab = [0]


# APPROACH USING LOG LIKELIHOOD
def approach_log_likelihood():
    # GOAL: Introduce log likelihood to maximize likelihood
    # equivalent to maximizing the log likelihood (because log is monotonic)
    # equivalent to minimizing negative log likelihood
    # equivalent to minimizing average negative log likelihood
    log_likelihood = 0
    n = 0
    for song in songs:
        for ch1, ch2 in zip(song, song[1:]):
            # probability of ch1, ch2 tuple
            prob = P[ch1, ch2]
            # log of that probability will return 0 if close to 1; else it will get more and more negative
            logprob = torch.log(prob)
            # we sum the log of all tuples per word following log principle: log(a*b*c) = log(a) + log(b) + log(c)
            log_likelihood += logprob
            # count the number of tuples we evaluated so we get the average log likelihood
            n += 1
            # print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")
    print(f"{log_likelihood=}")
    # nll = negative log likelihood (we can use it better if its positive)
    nll = -log_likelihood
    print(f"{nll=}")
    # average log likelihood per tuple (n)
    print(f"{nll/n}")


def gradient_descent_neural_network_approach():
    import torch.nn.functional as F

    # create training set of all bigrams (x, y)
    xs, ys = [], []
    for song in songs[1:]:
        for ch1, ch2 in zip(song, song[1:]):
            xs.append(ch1)
            ys.append(ch2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num_examples = xs.nelement()
    W = torch.randn((20, 20), generator=g, requires_grad=True)
    learning_rate = -12

    for _ in range(100):
        # forward pass
        xenc = F.one_hot(xs, num_classes=20).float()
        logits = xenc @ W  # log counts
        counts = (
            logits.exp()
        )  # we normalize the above result to have negative values below 1 and positive ones above 1
        # counts is similar to N matrix
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(num_examples), ys].log().mean()
        # print(loss.item())

        # backward pass
        W.grad = None
        loss.backward()

        # update
        W.data += learning_rate * W.grad

    # give predictions based on this model
    new_tab = [0]
    for _ in range(10):
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=20).float()
            logits = xenc @ W  # log counts
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)

            ix = torch.multinomial(
                probs, num_samples=1, replacement=True, generator=g
            ).item()
            new_tab.append(ix)
            if ix == 0:
                break

        print(new_tab)
        new_tab = [0]

    # nlls = torch.zeros(9)
    # for i in range(9):
    #     x = xs[i].item()
    #     y = ys[i].item()
    #     print("----------------")
    #     print(f"bigram example {i+1}: {x}{y} (indexes {x}{y})")
    #     print(f"input to the neural net: {x}")
    #     print(f"output probabilities from the neural net: {probs[i]}")
    #     print("label (actual next character):", y)
    #     p = probs[i, y]
    #     print("probability assigned by the net to the correct character: ", p.item())
    #     logp = torch.log(p)
    #     print("log likelihood:", logp.item())
    #     nll = -logp
    #     nlls[i] = nll


# show_occurrence_graph()
first_approach_efficient()
print("-----------------------")
# approach_log_likelihood()
gradient_descent_neural_network_approach()
"""

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
