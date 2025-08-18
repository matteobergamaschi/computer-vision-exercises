from io import BytesIO
import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt


with open("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\gallery_0.jpg", "rb") as f:
    bio = BytesIO(f.read())
bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
im_a = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
im_a = np.swapaxes(np.swapaxes(im_a, 0, 2), 1, 2)
im_a = im_a[::-1, :, :]  # from BGR to RGB

with open("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\gallery_1.jpg", "rb") as f:
    bio = BytesIO(f.read())
bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
im_b = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
im_b = np.swapaxes(np.swapaxes(im_b, 0, 2), 1, 2)
im_b = im_b[::-1, :, :]  # from BGR to RGB

# frame keypoints
# up left 137, 50
# down left 132, 191
# up right 339, 59
# down right 334, 199

# label keypoints
# 76, 130
# 76, 165
# 95, 130
# 95, 165

# angled frame keypoints
# 193,33
# 180,240
# 317,94
# 310,210

# angled label keypoints
# 80,145
# 75,213
# 116,146
# 113,209

srcPoints = np.array([[137, 50], [132, 191], [339, 59], [334, 199], [76, 130], [76, 165], [95, 130], [95, 165]])
dstPoints = np.array([[193, 33], [180, 240], [317, 94], [310, 210], [80, 145], [75, 213], [116,146], [113,209]])

H, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC)
im_a = im_a.transpose((1,2,0))
im_b = im_b.transpose((1,2,0))
im_out = cv2.warpPerspective(im_a, H, (im_b.shape[1], im_b.shape[0]))
plt.imshow(im_out)
plt.show()




