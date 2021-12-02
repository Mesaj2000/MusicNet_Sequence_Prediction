import numpy as np
import os
from glob import glob
from matplotlib import pyplot as plt
import torch

label_paths = glob("sequence_labels/*/*.npy")

label_counter = np.zeros(84)
adjusted_label_counter = np.zeros(84)
print(len(label_paths))

for path in label_paths:
    label = np.load(path)
    label_counter += (label > 0.5).astype(int)
    adjusted_label_counter += (label > 0.5).astype(int) * 2 - 1

total = label_counter.sum()
print(total)
label_counter /= total
adjusted_label_counter /= total

x = list(range(0, 84))
plt.scatter(x, adjusted_label_counter)
plt.show()