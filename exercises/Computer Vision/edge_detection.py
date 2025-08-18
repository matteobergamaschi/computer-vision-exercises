import torch
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def conv2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape  

    out_H = int(((H - kH)/1) + 1)
    out_W = int(((W - kW)/1) + 1)

    out = torch.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+kH, j:j+kW]
            out[i,j] = torch.sum(region * kernel)

    return out

im = cv.imread("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\\flowers.jpg", cv.IMREAD_GRAYSCALE)
im_tensor = torch.tensor(im, dtype=torch.uint8)

x_sobel = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]])
y_sobel = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]])

Gx = conv2d(im_tensor, x_sobel)
Gy = conv2d(im_tensor, y_sobel)

#calcolo il magnitude M che mi da la variazione tra il bordo e il resto dell'immagine
M = torch.sqrt(Gx**2 + Gy**2)

#calcolo l'angolo dei bordi in radianti, poi da trasformare in gradi in un intervallo: [0,179]
theta = torch.atan2(Gy, Gx)

#conversion
theta = torch.rad2deg(theta)
theta = torch.clamp(theta, 0.0 ,179.0).round().to(torch.uint8)
M = torch.clamp(M, 0.0, 255.0).round().to(torch.uint8)
S_torch = torch.full(M.shape, 255, dtype=torch.uint8)

#conversion into np.array for merge
H = theta.cpu().numpy()
S = S_torch.cpu().numpy()
V = M.cpu().numpy()

#creating the HSV image
hsv_image = cv.merge([H,S,V])

#RGB conversion
rgb_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)

#RGB
plt.imshow(rgb_image)
plt.show()
