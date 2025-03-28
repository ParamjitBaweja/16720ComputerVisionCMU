import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import LucasKanade
import cv2

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

p = np.zeros(2)

fig, ax = plt.subplots(1) 

# Display the image 
im = ax.imshow(seq[:,:,0]) 
carseqrects = np.zeros(shape= (seq.shape[2]-1, 4))
for i in range(0, seq.shape[2]-1):
# for i in range(0, 1):   
    # # ax.clear()
    # print(i)

    
    p = LucasKanade.LucasKanade(seq[:,:,i], seq[:,:,i+1], rect, threshold, num_iters, p)

    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    carseqrects[i,:] = rect
    # print(rect)

    # cv2.imshow("i",seq[:,:,i+1])
    # cv2.waitKey(0)
    # image = np.copy(seq[:,:,i+1])
    # image = np.ascontiguousarray(image, dtype='uint8')
    # image = image*255
    # cv2.imshow("frame", image)
    # # cv2.rectangle(image, (rect[0], rect[1]),(rect[2], rect[3]),  (255, 0, 0) , 2) 
    # # cv2.rectangle(image,(384,0),(510,128),(0,255,0),3)
    # cv2.waitKey(1)
    # plt.imshow(seq[:,:,100])


    # print(seq.shape)
    # ax = seq[:,:,100]


    # fig, ax = plt.subplots(1) 

    # Display the image 


    
    im.set_data(seq[:,:,i+1]) 

    points = [(rect[0], rect[1]),(rect[0],rect[3]),(rect[2], rect[3]),(rect[2],rect[1])]
    rectangle_draw = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
    # rectangle_draw.remove()
    c1= ax.add_patch(rectangle_draw)
    # plt.imshow(ax)
    plt.pause(0.05)
    frames = [0,99,199,299,399]
    if i in frames:
        print("ugh")
        input("frame, " + str(i))

    c1.remove()
    # plt.show(block = False)
# print(carseq)

np.save('carseqrects', carseqrects)
print(carseqrects.shape)