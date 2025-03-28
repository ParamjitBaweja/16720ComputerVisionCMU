import numpy as np
import cv2
import scipy
import math

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	#x1, x2 = matches - (nx2)
	#H2to1 is a 3x3 matrix
	#x y 1 0 0 0 -xu -yu -u
	#0 0 0 x y 1 -xv -yv -v
	A = []
	# print(x1.shape)
	# print(x2.shape())
	for i in range(x1.shape[0]):
	# for i in range(4):
		#flipping the [rows,col][y,x] format to x,y/u,v
		u = x1[i][0]
		v = x1[i][1]
		x = x2[i][0]
		y = x2[i][1] 
		point1 = np.array( [x, y, 1, 0, 0, 0, -(x*u), -(y*u), -u])
		point2 = np.array( [0, 0, 0, x, y, 1, -(x*v), -(y*v), -v])
		A.append(point1)
		A.append(point2)

	A = np.array(A)
	# print(A)
	# print(A.shape)

	U,S,Vh = np.linalg.svd(A)
	# print(H2to1)

	# result = np.linalg.eig(A)
	# print(result)
	H = Vh[-1, :] / Vh[-1,-1]
	# print(H)
	H2to1= np.reshape(H, (3,3))
	# print(x1[0])
	# print(x2[0])

	# print(H2to1)

	# x1_n  = np.zeros(shape=(x1.shape))
	# x2_n = np.zeros(shape=(x2.shape))
	# x1_n[:, 0] = x1 [:, 1]
	# x1_n[:, 1] = x1 [:, 0]
	# x2_n[:, 0] = x2 [:, 1]
	# x2_n[:, 1] = x2 [:, 0]
	# compare = cv2.findHomography( x1_n, x2_n)
	# print("opencv comparison", compare[0])


	
	return H2to1
	# return compare[0]


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	centroid_x1 = [ np.mean(x1[:,0]), np.mean(x1[:,1])]
	centroid_x2 = [ np.mean(x2[:,0]), np.mean(x2[:,1])]
	print("cent",centroid_x1)

	#Shift the origin of the points to the centroid
	# print("x1", x1)
	x1_n = x1 - centroid_x1
	x2_n = x2 - centroid_x2
	# print("x1_n", x1_n)


	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	# print( scipy.spatial.distance.cdist([centroid_x1], x1, metric='euclidean')[0])
	# distance_x1 = np.max(scipy.spatial.distance.cdist([centroid_x1], x1, metric='euclidean'))#[0])
	# distance_x2 = np.max(scipy.spatial.distance.cdist([centroid_x2], x2, metric='euclidean'))#[0])


	distance_x1 = np.max(np.linalg.norm(x1_n))
	distance_x2 = np.max(np.linalg.norm(x2_n))
	print("dist",distance_x1)
	print("dist",distance_x2)

	#Similarity transform 1
	t1 = np.array([ [ (np.sqrt(2)/distance_x1), 0, (  centroid_x1[0] * (-np.sqrt(2)/distance_x1)) ] , 
		[0, (np.sqrt(2)/distance_x1), (  centroid_x1[1] * (-np.sqrt(2)/distance_x1)) ], 
		[0,0,1] ])
	

	#Similarity transform 2
	t2 = np.array([ [ (np.sqrt(2)/distance_x2), 0, (  centroid_x2[0] * (-np.sqrt(2)/distance_x2)) ] , [0, (np.sqrt(2)/distance_x2), (  centroid_x2[1] * (-np.sqrt(2)/distance_x2)) ], [0,0,1] ] )
	# x1_normalized = x1*np.sqrt(2)/distance_x1
	# x2_normalized = x1*np.sqrt(2)/distance_x1

	x1_normalized = np.ones(shape=(np.shape(x1)[0], np.shape(x1)[1]+1))
	x2_normalized = np.ones(shape=(np.shape(x2)[0], np.shape(x2)[1]+1))
	x1_normalized[:, :2] = x1 
	x2_normalized[:, :2] = x2
	x1_normalized = np.transpose(np.matmul(t1, np.transpose(x1_normalized)))
	x2_normalized = np.transpose(np.matmul(t2, np.transpose(x2_normalized)))
	print("shape", x1_normalized.shape)

	#Compute homography
	H_norm = computeH(x1_normalized, x2_normalized)

	#Denormalization

	H2to1 = np.linalg.inv(t1) @ H_norm @ t2

	

	return H2to1

def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	# max_iters =1'

	try:

		max_inliers = 0
		bestH2to1 = np.zeros((3,3))
		for i in range(0,max_iters):
			# random_points = np.random.uniform(low = 0.0, high = len(locs1), size=(4))
			lenth = len(locs1)
			if lenth>=1000:
				lenth = 999
			random_points = np.random.choice(lenth, 4, replace=False)
			# random_points.astype('int')
			# print(locs1[random_points])
			H2to1 = computeH(locs1[random_points], locs2[random_points])

			# t = cv2.findHomography(locs1[random_points], locs2[random_points])
			# H2to1 = t[0]


			p1 = np.ones(shape=(np.shape(locs1)[0], np.shape(locs1)[1]+1))
			p2 = np.ones(shape=(np.shape(locs1)[0], np.shape(locs1)[1]+1))
			p1[:, :2] = locs1 
			p2[:, :2] = locs2
			# print(p2.shape)
			# print(H2to1.shape)

			n_p2 = np.transpose(np.matmul(H2to1, np.transpose(p2)))
			# n_p2 = n_p2/n_p2[:,2]
			# print(n_p2.shape)
			# print()

			# distances = np.sum((p1 - n_p2) ** 2) / p1.shape[0]
			inliers =0
			for i in range (0,len(p1)):
				distance = np.linalg.norm(p1[i]-n_p2[i])/100
				# distance = ( (p1[i][0] - n_p2[i][0])**2 / (p1[i][1]- n_p2[i][0])**2)**0.5
				# print(distance)
				if(distance<=inlier_tol):
					inliers = inliers+1
				
			# distances = np.linalg.norm(p1 - n_p2)
			# print(distances)

			# inliers = distances[distances <= inlier_tol].size

			# print("inl", inliers)
			try:
				if(inliers> max_inliers):
					# print("here")
					max_inliers = inliers
					bestH2to1 = H2to1
			except:
				continue
	except:
		inliers=0
		bestH2to1 = np.eye(3)
		return bestH2to1, inliers
		# print(len(inliers))
	# print(bestH2to1)
	# return np.linalg.inv(bestH2to1), inliers
	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography
	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template



	# # mask = np.zeros(shape=(template.shape[0], template.shape[1],3))
	# mask = template

	# #Warp mask by appropriate homography

	# # img_flat = np.reshape(img, (1, (img.shape[0]*img.shape[1])))
	# # print(img.shape)
	# for i in range(0,img.shape[0]):
	# 	for j in range(0,img.shape[1]):
	# 		n = H2to1 @ [[j],[i],[1]]
	# 		# print("index", (i,j))
	# 		# print("transform", n)
	# 		u = int( (n[0]/n[2]) )
	# 		v = int( (n[1]/n[2]) ) 
	# 		# print(img[i][j])
	# 		try:
	# 			mask [v] [u] = img[i][j] #/ 255.0
	# 		except:
	# 			continue
	# 		# print(mask[v][u])
	# # # cv2.imshow("mask", mask)
	# # # cv2.waitKey(0)
	# # # print(mask)
	# composite_img = mask

	template = template/255.0
	img = img/255.0
	# cover = cover / 255.0
	# print(np.max(template))
	# print(np.max(img))

	cover = cv2.warpPerspective(img, H2to1, dsize=(template.shape[1], template.shape[0]))
	# print(np.max(cover))
	# print("here")
	# print()

	mask = cv2.warpPerspective(np.ones(shape=img.shape), H2to1, dsize=(template.shape[1], template.shape[0]))
	# print(np.max(mask))

	# print(template.shape)
	# print(mask.shape)
	# composite_img= cv2.bitwise_or(template,mask)
	mask_inv = 1- mask
	composite_img = mask_inv * (template)
	composite_img = composite_img+ (cover)
	
	# mask = [1 if a_.any() > 0 else a_ for a_ in template]
	# mask = np.where(template > 0, 1, a)
	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	
	# composite_img = template
	return composite_img


