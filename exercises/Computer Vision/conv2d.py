import numpy as np
 

def gaussian_kernel(size, std):
    kernel = np.zeros((size, size))
    mean = size // 2

    for x in range(size):
        for y in range(size):
            kernel[x,y] = (1/(2*np.pi*std**2)*np.exp(-((x-mean)**2 + (y-mean)**2)/2*std**2))

    kernel /= np.sum(kernel)
    return kernel

def conv2d(image, kernel, stride):
    # so we have to implement manual 2d convolution
    # to do that we will use a random generated numpy array and then we will pass
    # 3x3 kernel on top of the image.

    H, W = image.shape
    kH, kW = kernel.shape  
    
    # output dimensions after convolution
    out_H = int(((H - kH)/stride) + 1)
    out_W = int(((W - kW)/stride) + 1)

    result = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i:i+kH, j:j+kW]
            result[i,j] = np.sum(region * kernel)

    return result


image = np.random.randint(1,10, (9, 9))


# we try first with a kernel that enhances vertical patterns
kernel = np.array([[1,0,-1],
                   [1,0,-1],
                   [1,0,-1]])

result = conv2d(image, kernel, 1)

# then we try again with a gaussian kernel to see the difference
g_kernel = gaussian_kernel(3, 1)

print(g_kernel)

result = conv2d(image, g_kernel, 3)
print(result)
