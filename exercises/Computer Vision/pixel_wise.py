import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\photo-1550853024-fae8cd4be47f.jpg", cv2.IMREAD_COLOR).swapaxes(0,2)
im_tensor = torch.from_numpy(im.astype(np.float32))

a,b = 2,0.5

im_transformed = (im_tensor.to(torch.float32) * a + b)
im_transformed = torch.clamp(im_transformed, 0.0, 255.0).round().to(torch.uint8)

plt.imshow(im_transformed.swapaxes(0,2))
plt.show()

