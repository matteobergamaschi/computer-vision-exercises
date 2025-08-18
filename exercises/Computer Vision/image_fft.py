import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\\bird.jpg", cv.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap='grey')
plt.title('Input Image' ) #, plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap='grey')
plt.title('Magnitude Spectrum') #, plt.xticks([]), plt.yticks([])
plt.show()