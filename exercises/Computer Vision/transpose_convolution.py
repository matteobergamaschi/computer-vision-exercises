import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
s = random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)

oH = (H - 1) * s + kH
oW = (W - 1) * s + kW

out = torch.zeros(n, oC, oH, oW, dtype=torch.float32)

for k in range(n):
    for oc in range(oC):
        for i in range(H):
            for j in range(W):
                for ic in range(iC):
                    out[k, oc, i*s:i*s+kH, j*s:j*s+kW] += input[k, ic, i, j] * kernel[ic, oc, :, :]


print(out)

# alternative sol
