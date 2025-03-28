import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
vanilla_rects = np.load("girlseqrects.npy")
rect = [280, 152, 330, 318]

rect0 = np.copy(rect)


p = np.zeros(2)

fig, ax = plt.subplots(1) 

im = ax.imshow(seq[:,:,0]) 

p_last = np.copy(p)

tn = seq[:,:,0]
girlseqrects_wcrt = np.zeros(shape= (seq.shape[2]-1, 4))
for i in range(1, seq.shape[2]-1):   

    p = LucasKanade(tn, seq[:,:,i], rect, threshold, num_iters, p_last)
    p = np.array(p)

    p_diff = np.array([rect[0]-rect0[0], rect[1]-rect0[1]])
    p_n_star = LucasKanade(seq[:,:,0], seq[:,:,i], rect0 , threshold, num_iters, p + p_diff)

    if np.linalg.norm(np.array(p) + np.array(p_diff) - np.array(p_n_star)) <= template_threshold:
        
        p_temp = np.array([p_n_star[0]-p_diff[0], p_n_star[1]-p_diff[1]])
        rect = np.array([rect[0]+p_temp[0], rect[1]+p_temp[1], rect[2]+p_temp[0], rect[3]+p_temp[1]])
        tn = np.copy(seq[:,:,i])
        p_last = np.array([0, 0])
    else:

        p_last = np.copy(p)
    
    im.set_data(seq[:,:,i]) 
    girlseqrects_wcrt[i,:] = rect
    points = [(rect[0], rect[1]),(rect[0],rect[3]),(rect[2], rect[3]),(rect[2],rect[1])]
    rectangle_draw = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
    c1= ax.add_patch(rectangle_draw)
    points_1 = [(vanilla_rects[i,:][0], vanilla_rects[i,:][1]),(vanilla_rects[i,:][0],vanilla_rects[i,:][3]),(vanilla_rects[i,:][2], vanilla_rects[i,:][3]),(vanilla_rects[i,:][2],vanilla_rects[i,:][1])]
    rectangle_draw_1 = patches.Polygon(points_1, linewidth=1, edgecolor='b', facecolor='none')
    c2= ax.add_patch(rectangle_draw_1)
    plt.pause(0.000001)
    frames = [1,19,39,59,79]
    if i in frames:
        input("frame, " + str(i))
    c1.remove()
    c2.remove()

np.save('girlseqrects-wcrt', girlseqrects_wcrt)
print(girlseqrects_wcrt.shape)