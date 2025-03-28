import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
from matplotlib import pyplot as plt

import multiprocessing
from multiprocessing import Pool, get_context
import os
import time

def run(img, i, opts):

	#Rotate Image
	print(i)
	img_rot = scipy.ndimage.rotate(img, (i*10))

	# cv2.imshow("G",img_rot)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, img_rot, opts)


	#Update histogram
	# matches_final = []
	# for i in range(i * 10):
	# 	matches_final.append(matches)
	return [len(matches), (i*10)]

	# pass # comment out when code is ready

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')

x = []
y = []

start = time.time()

cpu = multiprocessing.cpu_count()
pool = get_context("fork").Pool(cpu)
args = [(img, i, opts) for i in range(1,36)]
result = pool.starmap(run, args)
for r in result:
	y.append(r[0])
	x.append(r[1])

end = time.time()
print(end - start)



print(y)
print(x)
# plt.hist(x)

plt.bar(x, y)
plt.show()
#Display histogram















# import numpy as np
# import cv2
# from matchPics import matchPics
# from opts import get_opts
# import scipy
# from matplotlib import pyplot as plt

# opts = get_opts()
# #Q2.1.6
# #Read the image and convert to grayscale, if necessary
# img = cv2.imread('../data/cv_cover.jpg')

# x = []
# y = []
# for i in range(36):
# 	#Rotate Image
# 	print(i)
# 	img_rot = scipy.ndimage.rotate(img, (i*10))
	
# 	# cv2.imshow("G",img_rot)
# 	# cv2.waitKey(0)
# 	# cv2.destroyAllWindows()
# 	#Compute features, descriptors and Match features
# 	matches, locs1, locs2 = matchPics(img, img_rot, opts)


# 	#Update histogram
# 	x.append((i*10))
# 	y.append(len(matches))

# 	# pass # comment out when code is ready

# print(y)
# print(x)
# # plt.hist(y)
# # plt.show()
# #Display histogram

