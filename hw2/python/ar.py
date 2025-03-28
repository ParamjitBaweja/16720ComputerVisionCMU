import numpy as np
import cv2
#Import necessary functions

from matchPics import matchPics
from helper import plotMatches
from planarH import computeH_ransac, compositeH
from opts import get_opts

import loadVid

import multiprocessing
from multiprocessing import Pool, get_context
import os
import time

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


#Write script for Q3.1

opts = get_opts()
# cap = cv2.VideoCapture('../data/book.mov')
# cap2 = cv2.VideoCapture('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')
# count = 0

book_source = loadVid.loadVid('../data/book.mov')
print("clip 1 read")
print(book_source.shape)
ar_source = loadVid.loadVid('../data/ar_source.mov')
print("clip 2 read")
print(ar_source.shape)

# print(cap)

def run (cv_cover, frame, frame2, i, opts):
    try:
# for i in range(0,ar_source.shape[0]):
        print ("image ",i)

        # ret,frame = cap.read()
        # frame = np.array(book_source[i])
        # # ret2,frame2 = cap2.read()
        # frame2 = np.array(ar_source[i])

        # print(frame2.shape)

        cv_cover = cv2.resize(cv_cover, (287,360))
        # cv2.imshow("fig",cv_cover)
        # m_row = int(frame2.shape[0]/2)
        # m_col = int(frame2.shape[1]/2)

        # print(frame2.shape)
        # print(cv_cover.shape)

        # print(m_row)
        # print(int(m_row - (cv_cover.shape[0]/2) ))
        # print(frame2.shape)
        frame2 = frame2 [ : , 177: 463] 
        # frame2 = frame2[ int(m_row - (cv_cover.shape[0]/2) ) : int(m_row + (cv_cover.shape[0]/2)) ] [ int(m_col - (cv_cover.shape[1]/2) ) : int(m_col + (cv_cover.shape[1]/2))]
        # cv_cover.shape[0])
        # print(frame2.shape)

        frame2 = cv2.resize(frame2, (cv_cover.shape[1],cv_cover.shape[0]))
        

        matches, locs1, locs2 = matchPics(frame, cv_cover, opts)
        # if(len(matches)<4):
        #     print("very few matches")
        #     continue
        x1, x2 = reorder_locations(matches, locs1, locs2)
        H2to1, inliers = computeH_ransac(x1, x2, opts)

        # print (H2to1)
        book = compositeH(H2to1, frame, frame2)

        # cv2.imshow("norm",hp_cover_wrapped)


        return book
    except Exception as error:
        print("An error occurred:", error)
        # continue
    # cv2.imshow('book', book)
    # # cv2.imshow('source', frame2)

    # # cv2.imwrite("frame%d.jpg" % count, frame)
    # # count = count + 1
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

# cap.release()
# cv2.destroyAllWindows()

cpu = multiprocessing.cpu_count() - 1
pool = get_context("fork").Pool(cpu)
args = [(cv_cover,  book_source[i], ar_source[i], i, opts) for i in range(0,ar_source.shape[0]-1)]
result = pool.starmap(run, args)



# writer = cv2.VideoWriter('../data/try.avi', 
#                          cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'),
#                          10, (640,480),True)

# video_filename = '../data/try.mp4'
video_filename = '../result/ar.avi'
frame_rate = 10
frame_size = (640,480)
# fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fourcc = cv2.VideoWriter.fourcc('m','p','4','v')
writer = cv2.VideoWriter(
  video_filename,
  fourcc,
  frame_rate,
  frame_size
)
    
print("result",np.shape(result))
for j in range(0, len(result)):
    # print(j)
    # print("hi")
    # print(result[j])
    # frame = np.array(result[j] , dtype='uint8')
    # frame = frame*255
    frame = result[j]*255
    # print("frae",np.shape(frame))
    # cv2.imshow('Frame', frame)
    # print(np.max(frame))
    writer.write(frame.astype(np.uint8))

  
    # Press S on keyboard 
    # to stop the process
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break


