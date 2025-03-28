import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

# import multiprocessing
# from multiprocessing import Pool, get_context
# import os


# import time
# def run(img,opts):
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	locs = corner_detection(img, opts.sigma)
# 	desc, locs = computeBrief(img, locs)
# 	return desc, locs

# def matchPics(I1, I2, opts):
# 	#I1, I2 : Images to match
# 	#opts: input opts
# 	start = time.time()
# 	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
# 	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

# 	# #Convert Images to GrayScale

# 	I1_g = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
# 	I2_g = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)


	
# 	cpu = multiprocessing.cpu_count()
# 	pool = get_context("fork").Pool(cpu)
# 	args = [(I1, opts),(I2, opts)]
# 	result = pool.starmap(run, args)


# 	matches = briefMatch(result[0][0], result[1][0], ratio)
# 	locs1 = result[0][1]
# 	locs2 = result[1][1]

# 	end = time.time()
# 	print("Time:",end - start)

# 	return matches, locs1, locs2



def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

	# #Convert Images to GrayScale

	I1_g = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2_g = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)


	# cv2.imshow("G",I1_g)
	# cv2.imshow("G2",I2_g)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	
	#Detect Features in Both Images
	I1_locs = corner_detection(I1_g, sigma)
	I2_locs = corner_detection(I2_g, sigma)
	
	#Obtain descriptors for the computed feature locations
	I1_desc, I1_locs = computeBrief(I1_g, I1_locs)
	I2_desc, I2_locs = computeBrief(I2_g, I2_locs)
	
	# cpu = multiprocessing.cpu_count()
	# pool = get_context("fork").Pool(cpu)
	# args = [(I1, opts),(I2, opts)]
	# result = pool.starmap(run, args)

	#Match features using the descriptors
	# print(result)
	# matches = briefMatch(result[0][0], result[1][0], ratio)
	# locs1 = result[0][1]
	# locs2 = result[1][1]

	matches = briefMatch(I1_desc, I2_desc, ratio)
	locs1 = I1_locs
	locs2 = I2_locs

	return matches, locs1, locs2
