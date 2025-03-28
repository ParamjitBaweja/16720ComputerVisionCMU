import numpy as np
import cv2
#Import necessary functions

from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac, compositeH
from opts import get_opts
import time
import loadVid

import matplotlib.pyplot as plt 




#Write script for Q3.1

opts = get_opts()
cap = cv2.VideoCapture('../data/book.mov')
cap2 = cv2.VideoCapture('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')
# count = 0


prev_frame_time = 0
new_frame_time = 0


count =-1
# print(cap)

# book_source = loadVid.loadVid('../data/book.mov')
# print("clip 1 read")
# print(book_source.shape)
# ar_source = loadVid.loadVid('../data/ar_source.mov')
# print("clip 2 read")
# print(ar_source.shape)
for i in range(0,511):
    # print(i)
    count = count + 1
    # print(book_source[i].shape)
    # print(ar_source[i].shape)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    # print(fps)
    ret,frame = cap.read()
    # frame = np.array(book_source[i])
    ret2,frame2 = cap2.read()
    # frame2 = np.array(ar_source[i])

    # print(frame)

    cv_cover = cv2.resize(cv_cover, (287,360))
 
    frame2 = frame2 [ : , 177: 463] 

    frame2 = cv2.resize(frame2, (cv_cover.shape[1],cv_cover.shape[0]))
    
    if count%5 == 0:
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(frame,None)
        kp2, des2 = orb.detectAndCompute(cv_cover,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        x1 = np.array([kp1[mat.queryIdx].pt for mat in matches[:10]] )
        x2 = np.array([kp2[mat.trainIdx].pt for mat in matches[:10]])
        # print(frame)
        # try:

        # sift = cv2.SIFT_create()
        # # find the keypoints and descriptors with ORB
        # kp1, des1 = sift.detectAndCompute(frame,None)
        # kp2, des2 = sift.detectAndCompute(cv_cover,None)

        # # 
        # bf = cv2.BFMatcher()
        # # Match descriptors.
        # # matches = bf.match(des1,des2)
        # matches = bf.knnMatch(des1,des2, k=2)


        # good = []
        # good_without_list = []

        # for m, n in matches:
        #     if m.distance < 0.5 * n.distance:
        #         good.append([m])
        #         good_without_list.append(m)


        # x1 = np.array([kp1[mat.queryIdx].pt for mat in good_without_list[:]] )
        # x2 = np.array([kp2[mat.trainIdx].pt for mat in good_without_list[:]])


        t = cv2.findHomography(x2, x1)
        H2to1 = t[0]
        H2to1 = H2to1/H2to1[2,2]

    # H2to1, inliers = computeH_ransac(x1, x2, opts)

    # print (H2to1)
    book = compositeH(H2to1, frame, frame2)


    fps = "FPS:"+str(fps)
  
    # putting the FPS count on the frame
    cv2.putText(book, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('book', book) 
    # cv2.waitKey(0)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

####################################################################################################
## code to run it real time with SIFT
####################################################################################################


# import numpy as np
# import cv2
# #Import necessary functions

# from matchPics import matchPics
# from helper import plotMatches
# from planarH import computeH_ransac, compositeH
# from opts import get_opts
# import time
# import loadVid

# import matplotlib.pyplot as plt 




# #Write script for Q3.1

# opts = get_opts()
# cap = cv2.VideoCapture('../data/book.mov')
# cap2 = cv2.VideoCapture('../data/ar_source.mov')
# cv_cover = cv2.imread('../data/cv_cover.jpg')
# # count = 0


# prev_frame_time = 0
# new_frame_time = 0


# count =0 
# # print(cap)

# # book_source = loadVid.loadVid('../data/book.mov')
# # print("clip 1 read")
# # print(book_source.shape)
# # ar_source = loadVid.loadVid('../data/ar_source.mov')
# # print("clip 2 read")
# # print(ar_source.shape)
# for i in range(0,511):
#     # print(i)
#     count = count + 1
#     # print(book_source[i].shape)
#     # print(ar_source[i].shape)
#     new_frame_time = time.time()
#     fps = 1/(new_frame_time-prev_frame_time)
#     prev_frame_time = new_frame_time
#     fps = int(fps)
#     # print(fps)
#     ret,frame = cap.read()
#     # frame = np.array(book_source[i])
#     ret2,frame2 = cap2.read()
#     # frame2 = np.array(ar_source[i])

#     # print(frame)

#     cv_cover = cv2.resize(cv_cover, (287,360))
 
#     frame2 = frame2 [ : , 177: 463] 

#     frame2 = cv2.resize(frame2, (cv_cover.shape[1],cv_cover.shape[0]))
    

#     orb = cv2.ORB_create()
#     # find the keypoints and descriptors with ORB
#     # kp1, des1 = orb.detectAndCompute(frame,None)
#     # kp2, des2 = orb.detectAndCompute(cv_cover,None)
#     # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     # matches = bf.match(des1,des2)
#     # matches = sorted(matches, key = lambda x:x.distance)
#     # x1 = np.array([kp1[mat.queryIdx].pt for mat in matches[:10]] )
#     # x2 = np.array([kp2[mat.trainIdx].pt for mat in matches[:10]])


#     # print(frame)
#     # try:

#     sift = cv2.SIFT_create()
#     # find the keypoints and descriptors with ORB
#     kp1, des1 = sift.detectAndCompute(frame,None)
#     kp2, des2 = sift.detectAndCompute(cv_cover,None)

#     # 
#     bf = cv2.BFMatcher()
#     # Match descriptors.
#     # matches = bf.match(des1,des2)
#     matches = bf.knnMatch(des1,des2, k=2)


#     good = []
#     good_without_list = []

#     for m, n in matches:
#         if m.distance < 0.5 * n.distance:
#             good.append([m])
#             good_without_list.append(m)


#     x1 = np.array([kp1[mat.queryIdx].pt for mat in good_without_list[:]] )
#     x2 = np.array([kp2[mat.trainIdx].pt for mat in good_without_list[:]])


#     t = cv2.findHomography(x2, x1)
#     H2to1 = t[0]
#     H2to1 = H2to1/H2to1[2,2]

#     # H2to1, inliers = computeH_ransac(x1, x2, opts)

#     # print (H2to1)
#     book = compositeH(H2to1, frame, frame2)


#     fps = "FPS:"+str(fps)
  
#     # putting the FPS count on the frame
#     cv2.putText(book, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
#     cv2.imshow('book', book)
#     cv2.waitKey(1)
#     # cv2.waitKey(0)
#     # if cv2.waitKey(10) & 0xFF == ord('q'):
#     #     break

# cv2.destroyAllWindows()
