import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 5)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand(n, iC, T, H, W)

oT = ((T-kT)//s) + 1
oH = ((H-kH)//s) + 1
oW = ((W-kW)//s) + 1

out = torch.zeros(n, iC, oT, oH, oW, dtype=torch.float32)

for l in range(iC):
    for t in range(oT):
        for i in range(oH):
            for j in range(oW):
                region = input[:, l, t*s:t*s+kT, i*s:i*s+kH, j*s:j*s+kW]
                out[:, l, t, i, j] = torch.max(region)