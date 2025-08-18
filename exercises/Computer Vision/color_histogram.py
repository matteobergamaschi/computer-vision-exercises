import torch
import cv2
import matplotlib.pyplot as plt

q_bins = 100

def color_norm_hist(channel, nbin):
    '''
    H, W = channel.shape

    for i in range(H):
        for j in range(W):
            b = int((channel[i,j]*nbin)//256)
            hist[b] += 1
    '''
    
    '''
    channel = channel.to(torch.float32)
    hist, bin_edges = torch.histogram(channel, nbin)
    hist /= hist.sum()
    print(hist.sum())
    return hist
    '''
    
    channel = channel.to(torch.int32)
    q_channel = (channel * nbin) // 256
    hist = torch.bincount(q_channel.flatten(), minlength=nbin)
    hist = hist.to(torch.float32)
    return hist

im = cv2.imread("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\photo-1550853024-fae8cd4be47f.jpg", cv2.IMREAD_COLOR).swapaxes(0,2)

im_tensor = torch.tensor(im, dtype=torch.uint8)

r_hist = color_norm_hist(im_tensor[0], q_bins)
g_hist = color_norm_hist(im_tensor[1], q_bins)
b_hist = color_norm_hist(im_tensor[2], q_bins)

full_hist = torch.cat((r_hist, g_hist, b_hist), 0)
full_hist /= full_hist.sum()
print(full_hist)


plt.hist(r_hist, bins=q_bins, label='R', color='red')
plt.hist(g_hist, bins=q_bins, label='G', color='blue')
plt.hist(b_hist, bins=q_bins, label='B', color='green')

plt.show()