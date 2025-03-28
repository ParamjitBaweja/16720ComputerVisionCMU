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

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

hp_cover_big = cv2.resize(hp_cover, (cv_cover.shape[1],cv_cover.shape[0]))

# print(cv_cover.shape)
# print(hp_cover.shape)
# print(hp_cover_big.shape)
# cv2.imshow("cover", hp_cover_big)
# cv2.waitKey(0)

# matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

matches, locs1, locs2 = matchPics(cv_desk, cv_cover,  opts)

# print(matches[0])
# computeH()
x1, x2 = reorder_locations(matches, locs1, locs2)

# H2to1 = computeH_norm(x1, x2)
# np.random.seed(None)
H2to1, inliers = computeH_ransac(x1, x2, opts)
# t = cv2.findHomography( x1,x2)
# H2to1 = t[0]
# print(H2to1)
# print("here")

# print("umm", (cv_desk.shape[1], cv_desk.shape[0]))
#hp_cover_wrapped = cv2.warpPerspective(hp_cover_big, H2to1, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
hp_cover_wrapped = compositeH(H2to1, cv_desk, hp_cover_big)
cv2.imshow("norm",hp_cover_wrapped)
# cv = cv2.warpPerspective(hp_cover_big, H2to1, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
# cv2.imshow("cv",cv)

plotMatches(cv_desk, cv_cover, matches, locs1, locs2)
# cv2.waitKey(0)
