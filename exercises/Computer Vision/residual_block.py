import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=None)
        self.batch_norm1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=None)
        self.batch_norm2 = nn.BatchNorm2d(planes)

        if inplanes != planes or stride != 1:
            self.projection = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=None),
                nn.BatchNorm2d(planes)
            )
        else:
            self.projection = nn.Identity()


    def forward(self, x):
        G = self.projection(x)
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        out = self.relu(out + G)
        return out


im = cv.imread("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\\flowers.jpg", cv.IMREAD_GRAYSCALE)
im_tensor = torch.tensor(im, dtype=torch.float32)
im_tensor = im_tensor.unsqueeze(0).unsqueeze(1)

print(im_tensor.shape)
block = ResidualBlock(1,1,2)
res_tensor = block.forward(im_tensor)

plt.imshow(im_tensor.squeeze(0).squeeze(0))
plt.show()

plt.imshow(res_tensor.detach().numpy().squeeze(0).squeeze(0))
plt.show()



