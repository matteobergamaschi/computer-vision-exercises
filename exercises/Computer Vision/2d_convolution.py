import torch
import matplotlib.pyplot as plt
import random


def conv2d(im, kernel):
    n, iC, H, W = im.shape
    oC, iC, kH, kW = kernel.shape # guarda le dimensioni e si capisce perchè metti o quando scorri davanti
    stride = 1

    out_H = int(((H - kH)/stride) + 1)
    out_W = int(((W - kW)/stride) + 1)

    out = torch.zeros(n, oC, out_H, out_W, dtype=torch.float32)
    
    for o in range(oC):
        for i in range(out_H):
            for j in range(out_W):
                region = im[:, : ,i:i+kH, j:j+kW]
                out[:, o, i, j] = torch.sum(region * kernel[o,:,:,:]).dim(1,2,3)
                    
    return out

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, H, W, dtype=torch.float32)
kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)

out = conv2d(input, kernel)
print(out.shape)

# without for loops:

# questo primo codice crea praticamente una view per le caselle della convoluzione
# quindi in sostanza utilizzando questa view, sto creando le varie finestre, come farei con i for

'''
image_view = torch.as_strided(input, (n, iC, H - kH + 1, kH, W - kW + 1, kW), (iC * H * W, H * W, W, W, 1, 1))
'''

# per spiegarla praticamente all'inizio definisco la shape che voglio ottenere dalla view, e poi uso lo stride
# per visualizzare tutte le shape, spostandomi solamente di come vedo la memoria ma di fatto non usando for

# quindi in questo caso, quello che voglio vedere è il batch di immagini, ognuna con il numero di canali, e di
# queste voglio solamente vedere i pezzi delimitati dallo stride

# in questa parte invece, cambio la kernel_view in modo tale che sia compatibile con la image_view per poi 
# fare le operazioni punto punto con il broadcasting

'''
kernel_view = torch.unsqueeze(kernel, 0)       # (1, oC, iC, kH, kW)
kernel_view = torch.unsqueeze(kernel_view, 3)   # (1, oC, iC, 1, kH, kW)
kernel_view = torch.unsqueeze(kernel_view, 5)   # (1, oC, iC, 1, kH, 1, kW)
image_view = torch.unsqueeze(image_view, 1)     # (n, 1, iC, H - kH + 1, kH, W - kW + 1, kW)
'''

# kernel_view = kernel.unsqueeze(0).unsqueeze(3).unsqueeze(5)

'''
out = image_view * kernel_view
out = torch.sum(out, (2,4,6))
'''

# final without for loops

image_view = torch.as_strided(input, (n, iC, H - kH + 1, kH, W - kW + 1, kW), (iC * H * W, H * W, W, W, 1, 1))
kernel_view = kernel.unsqueeze(0).unsqueeze(3).unsqueeze(5)
image_view = torch.unsqueeze(image_view, 1)
out = image_view * kernel_view
out = torch.sum(out, (2,4,6))
