import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import cv2

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')


# p = np.zeros(2)

# fig, ax = plt.subplots(1) 

# im = ax.imshow(seq[:,:,0]) 

# for i in range(0, seq.shape[2]-1):    
# for i in range(0, 1):  
for i in [29, 59, 89, 119]:
    mask = SubtractDominantMotion (seq[:,:,i], seq[:,:,i+1], threshold, num_iters, tolerance)
    image2 = np.copy(seq[:,:,i+1])
    # print(image2.shape)
    image2_3_channel = cv2.merge((image2, image2, image2))
    # print(image2_3_channel.shape)
    mask_3_channel = cv2.merge((mask, mask, mask))

    mask_3_channel [mask==1]=(1, 0, 0)
    # img = image2_3_channel * mask_3_channel
    img = cv2.bitwise_or(image2_3_channel, mask_3_channel)
    # img = mask * image2
    plt.imshow(img)
    plt.show()
    
    # p = LucasKanade.LucasKanade(seq[:,:,i], seq[:,:,i+1], rect, threshold, num_iters, p)
    # rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    # im.set_data(img) 
    print(i)
    # points = [(rect[0], rect[1]),(rect[0],rect[3]),(rect[2], rect[3]),(rect[2],rect[1])]
    # rectangle_draw = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
    # c1= ax.add_patch(rectangle_draw)
    # plt.pause(1)
    # plt.show()
    # c1.remove()
