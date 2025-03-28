import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

p = np.zeros(2)

fig, ax = plt.subplots(1) 

im = ax.imshow(seq[:,:,0]) 

girlseqrects = np.zeros(shape= (seq.shape[2]-1, 4))

for i in range(0, seq.shape[2]-1):    
    p = LucasKanade.LucasKanade(seq[:,:,i], seq[:,:,i+1], rect, threshold, num_iters, p)
    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    girlseqrects[i,:] = rect
    im.set_data(seq[:,:,i+1]) 
    points = [(rect[0], rect[1]),(rect[0],rect[3]),(rect[2], rect[3]),(rect[2],rect[1])]
    rectangle_draw = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
    c1= ax.add_patch(rectangle_draw)
    plt.pause(0.0001)
    frames = [0,19,39,59,79]
    if i in frames:
        input("frame, " + str(i))
    c1.remove()
np.save('girlseqrects', girlseqrects)