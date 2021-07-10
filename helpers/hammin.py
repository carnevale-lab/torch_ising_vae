import numpy as np
from mi3gpu.utils.seqtools import histsim
from mi3gpu.utils.seqload import loadSeqs, writeSeqs
import sys
import matplotlib.pyplot as plt

def arr_to_str(in_arr):
    strs = []
    for i in range(len(in_arr)):
        strout = ""
        for j in range(len(in_arr[0])):
            strout+= str(in_arr[i][j])
        strs.append(strout)

    return np.array(strs)

# original = np.load("data/ising.npy")
# original = original.astype(int)
# predicted = np.load("generated/genSeqs.npy")
# predicted = predicted.astype(int)

# print(original.shape)
# print(predicted.shape)

# orig = arr_to_str(original)
# pred = arr_to_str(predicted)

# with open("orig.txt", "w") as f:
#     for line in orig:
#         f.write(line + '\n')
# f.close()

# with open("pred.txt", "w") as g:
#     for line in pred:
#         g.write(line + '\n')
# g.close()

tracker = dict()

seqs = loadSeqs("orig.txt", names="01")[0][0:]
h = histsim(seqs).astype(float)
h = h/np.sum(h)
rev_h = h[::-1]

seqs1 = loadSeqs("pred.txt", names="01")[0][0:]
h1 = histsim(seqs1).astype(float)
h1 = h1/np.sum(h1)
rev_h1 = h1[::-1]


fig, ax = plt.subplots()


ax.plot(rev_h, label="Original")
ax.plot(rev_h1, label="Predicted")

plt.legend(loc="best")

plt.savefig("test1.png")
