import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
from planarH import computeH, computeH_norm, computeH_ransac
import scipy


opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')


matches, locs1, locs2 = matchPics(cv_desk, cv_cover,  opts)

plotMatches(cv_desk, cv_cover, matches, locs1, locs2)
