import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)

input = torch.rand((n, iC, H, W), dtype=torch.float32)

oH = ((H - kH) // s) + 1
oW = ((W - kW) // s) + 1

out = torch.zeros(n, iC, oH, oW, dtype=torch.float32)

# non devo scorrere con lo stride che varia di s (range(0, oH, s)) perchè ho il 
# valore i e j che deve scorrere di 1 nell'output
for k in range(n):
    for l in range(iC):
        for i in range(oH):
            for j in range(oW):
                region = input[k, l, i*s:i*s+kH, j*s:j*s+kW]
                out[k, l, i, j] = torch.max(region)

print(out.shape)