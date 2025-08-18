import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_w(hist, start, end):
    res = 0
    for i in range(start, end):
        res += hist[i]
    return res

def calculate_mean(hist, start, end, w):
    res = 0
    for i in range(start, end):
        res += (hist[i]*i)/w
    return res

def inter_class_variance(hist, t):
    w_1 = calculate_w(hist, 0,t+1)
    w_2 = calculate_w(hist, t+2, 256)
    mu_1 = calculate_mean(hist, 0,t+1,w_1)
    mu_2 = calculate_mean(hist, t+2,256,w_2)

    return w_1 * w_2 * (mu_1-mu_2)**2


# change it t: and :t
im = cv2.imread("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\photo-1550853024-fae8cd4be47f.jpg", cv2.IMREAD_GRAYSCALE)

im_tensor = torch.tensor(im, dtype=torch.float32)
hist, _ = torch.histogram(im_tensor, bins=256)
hist /= hist.sum()

new = 0
fin_t = 0
for i in range(256):
    var = inter_class_variance(hist, i)
    if var > new:
        new = var
        fin_t = i

print(new)
print(i)


out = (im_tensor > fin_t).to(torch.uint8) * 255

plt.imshow(out, cmap='gray')
plt.show()

