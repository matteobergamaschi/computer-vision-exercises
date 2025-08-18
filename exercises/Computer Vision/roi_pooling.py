import random
import torch
import math
import torch.nn.functional as F

n = random.randint(1, 3)
C = random.randint(10, 20)
H = random.randint(5, 10)
W = random.randint(5, 10)
oH = random.randint(2, 4)
oW = random.randint(2, 4)
L = random.randint(2, 6)
input = torch.rand(n, C, H, W)
boxes = [torch.zeros(L, 4) for _ in range(n)]

for i in range(n):
  boxes[i][:, 0] = torch.rand(L) * (H-oH)       # y
  boxes[i][:, 1] = torch.rand(L) * (W-oW)       # x
  boxes[i][:, 2] = oH + torch.rand(L) * (H-oH)  # w
  boxes[i][:, 3] = oW + torch.rand(L) * (W-oW)  # h

  boxes[i][:,2:] += boxes[i][:,:2]
  boxes[i][:,2] = torch.clamp(boxes[i][:,2], max=H-1)
  boxes[i][:,3] = torch.clamp(boxes[i][:,3], max=W-1)
output_size = (oH, oW)

#boxes[i] =
#[
#  [y_min, x_min, y_max, x_max],
#  [y_min, x_min, y_max, x_max],
#  [y_min, x_min, y_max, x_max],
#  ...
#]

out = torch.zeros(n, L, C, oH, oW)

for k in range(n):
  for l in range(L):
    y_min, x_min, y_max, x_max = boxes[k][l].round().to(torch.int)

    for i in range(oH):
      y_start, y_end = math.floor(y_min+i*(y_max - y_min +1)/oH), math.ceil(y_min + (i+1) * (y_max - y_min +1)/oH)
        
      for j in range(oW):
        x_start, x_end = math.floor(x_min+j*(x_max - x_min +1)/oW), math.ceil(x_min + (j+1) * (x_max - x_min +1)/oW)
        out[k,l,:,i,j] = input[k, :, y_start:y_end, x_start:x_end].max(dim=-1)[0].max(dim=-1)[0]
          
          

