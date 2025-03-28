import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
# from q2_1_4 import reorder_locations
from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac, compositeH, computeH_norm

def reorder_locations (matches, locs1,locs2):
    
    x1 = []
    x2 = []
    for i in range (len(matches)):
        x1.append(locs1[matches[i][0]])
        x2.append(locs2[matches[i][1]])

    x1 = np.array(x1)
    x2 = np.array(x2)
    
    x1_n  = np.zeros(shape=(x1.shape))
    x2_n = np.zeros(shape=(x2.shape))
    x1_n[:, 0] = x1 [:, 1]
    x1_n[:, 1] = x1 [:, 0]
    x2_n[:, 0] = x2 [:, 1]
    x2_n[:, 1] = x2 [:, 0]
    return x1_n, x2_n

#Write script for Q2.2.4
opts = get_opts()

# pano_left = cv2.imread('../data/pano_left.jpg')
# pano_right = cv2.imread('../data/pano_right.jpg')

pano_left = cv2.imread('../data/pano_3.jpg')
pano_right = cv2.imread('../data/pano_1.jpg')


matches, locs1, locs2 = matchPics(pano_left, pano_right,  opts)


x1, x2 = reorder_locations(matches, locs1, locs2)

H2to1,inliers = computeH_ransac(x1, x2, opts)


img = cv2.warpPerspective(pano_right, H2to1, dsize=(pano_left.shape[1]+1500, pano_left.shape[0]))
cv2.imshow("cv2",img)

for i in range(0,pano_left.shape[0]):
    for j in range(0,pano_left.shape[1]):
        img[i][j] = pano_left[i][j]
cv2.imshow("overlay",img)
# cv = cv2.warpPerspective(hp_cover_big, H2to1, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
# cv2.imshow("cv",cv)

plotMatches(pano_left, pano_right, matches, locs1, locs2)
cv2.waitKey(0)
