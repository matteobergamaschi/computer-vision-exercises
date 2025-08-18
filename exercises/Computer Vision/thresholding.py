import torch
import cv2
import matplotlib.pyplot as plt

im = cv2.imread("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\images.jpg",cv2.IMREAD_GRAYSCALE)
im_tensor = torch.tensor(im, dtype=torch.uint8)
T = 100

out = (im_tensor > T).to(torch.uint8) * 255

'''
# my implementation
H,W = im.shape
out = torch.zeros((H,W), dtype=torch.uint8)
for i in range(H):
    for j in range(W):
        if im[i,j] < T:
            out[i,j] = 255
        else:
            out[i,j] = 0
'''

plt.imshow(out, cmap='gray')
plt.show()

