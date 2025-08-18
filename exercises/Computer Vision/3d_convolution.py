import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, T, H, W)
kernel = torch.rand(oC, iC, kT, kH, kW)
bias = torch.rand(oC)

oT = ((T+2*0-1*(kT-1) - 1 )// 1) + 1
oH = ((H+2*0-1*(kH-1) - 1 )// 1) + 1
oW = ((T+2*0-1*(kW-1) - 1 )// 1) + 1

out = torch.zeros(n, oC, oT, oH, oW, dtype=torch.float32)


for o in range(oC):
    for t in range(oT):
        for i in range(oH):
            for j in range(oW):
                region = input[:,:,t:t+kT, i:i+kH, j:j+kW]
                out[:, o ,t , i , j] = torch.sum(region*kernel[o, :, :, :, :], dim=(1,2,3,4)) + bias[o]

