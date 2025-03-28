import numpy as np
import util
"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''

def eightpoint(pts1, pts2, M):
    x_l = pts1[:,0]/M
    y_l = pts1[:,1]/M
    x_r = pts2[:,0]/M
    y_r = pts2[:,1]/M

    # print("hello")

    P = np.vstack((x_r*x_l, y_l*x_r, x_r, x_l*y_r, y_r*y_l, y_r, x_l, y_l, np.ones(x_l.shape)))
    # P = P.T
    # print (P.shape)
      
    u, s, vh = np.linalg.svd( P @ P.T )
    # print(u.shape, s.shape, vh.shape)
    F8 = vh[-1, :]
    # print(F8)
    F8 = np.reshape(F8, (3, 3))

    
    F8 = util.refineF(F8, pts1/M, pts2/M)
    # F8 = util._singularize(F8)
    T = np.array([ [1/M,0,0], [0,1/M,0], [0,0,1] ])
    

    F8 = T.T @ F8 @ T
    # print(F8)
    return F8

    


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K1.T @ F @ K2
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # print(C1, pts1, C2, pts2)
    # data = np.load('../data/some_corresp.npz')
    # print("helppp")
    # intrinsics = np.load('../data/intrinsics.npz')
    # print(intrinsics['K1'])
    # essentialMatrix (K1, K2, )
    # print(intrinsics['K1'])
    # print(intrinsics.shape)
    # Replace pass by your implementation
    # return 0,1
    # C12 = np.array([ C1[0, :] - C1[2, :], C1[1, :] - C1[2, :], C2[0, :] - C2[2, :],  C2[1, :] - C2[2, :]])
    # A = []
    # for i in range(0, 3):
    #     print(C12[0][i])
    #     r1 = np.array(C12[0][i]*x_1)
    #     r2 = np.array(C12[1][i]*y_1)
    #     r3 = np.array(C12[2][i]*x_2)
    #     r4 = np.array(C12[3][i]*y_2)
    #     A.append(np.array([ r1,r2,r3,r4]))
    # A = np.array(A)
    # print(A.shape)



    x_1 = pts1[:,0]
    y_1 = pts1[:,1]
    x_2 = pts2[:,0]
    y_2 = pts2[:,1]
    C1 = np.array(C1)
    C2 = np.array(C2)

    W = []
    for i in range (0, x_1.shape[0]):
        # print(x_1[i])
        A = np.array([ C1[0, :] - C1[2, :]*x_1[i], 
                        C1[1, :] - C1[2, :]*y_1[i], 
                        C2[0, :] - C2[2, :]*x_2[i],  
                        C2[1, :] - C2[2, :]*y_2[i]])
        # print(A)
        _, _, vh = np.linalg.svd(A)
        # print(vh)

        W.append(vh[-1, :]/ vh[-1, -1])
    W = np.array(W)
    # print(W)

    error = 0
    for i in range (0, x_1.shape[0]):
        projection1 = C1 @ W[i, :].T
        projection2 = C2 @ W[i, :].T
        projection1 = projection1.T / projection1[-1]
        projection2 = projection2.T / projection2[-1]
        error += np.sum((projection1[:2]-pts1[i])**2 + (projection2[:2]-pts2[i])**2)
    return W[:, 0:3], error

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation

    window_size = 20
    x1 = round(x1)
    y1 = round(y1)
    patch_1 = patch1 = im1[y1 - round(window_size/2):y1+ round(window_size/2), x1- round(window_size/2):x1+ round(window_size/2)]

    sy, sx, _ = im2.shape
    l = F @ np.array([[x1], [y1], [1]])

    s = np.sqrt(l[0]**2+l[1]**2)

    l = l/s

    if l[0] != 0:
        ye = sy-1
        ys = 0
        xe = -(l[1] * ye + l[2])/l[0]
        xs = -(l[1] * ys + l[2])/l[0]
    else:
        xe = sx-1
        xs = 0
        ye = -(l[0] * xe + l[2])/l[1]
        ys = -(l[0] * xs + l[2])/l[1]

    num_points = 50000
    x_values = np.linspace(xs, xe, num_points)
    y_values = np.linspace(ys, ye, num_points)
    points = np.column_stack((x_values, y_values))
    points = points.astype(int)
    max_x, max_y = im2.shape[1], im2.shape[0]
    points[:, 0] = np.clip(points[:, 0], round(window_size/2), max_x - round(window_size/2)-1)
    points[:, 1] = np.clip(points[:, 1], round(window_size/2), max_y - round(window_size/2)-1)
    # print(points)
    # print(points.shape)
    minimum_dist = None
    new_points = []
    for i in range (0,points.shape[0]):
         patch_2 = im2[points[i, 1] - round(window_size/2):points[i, 1]+ round(window_size/2), points[i, 0]- round(window_size/2):points[i, 0]+ round(window_size/2)]
         dist = np.sum((patch_1-patch_2)**2)
        #  print(dist)
         if minimum_dist is None or dist < minimum_dist:
            minimum_dist = dist
            # print("d",dist)
            new_points = points[i]

    return new_points[0], new_points[1]
    pass

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    maximum_inliers = 0
    F = 0
    final_inliers = 0

    for i in range(0,nIters):
        print (i)
        lenth = 110
        # print(pts1.shape)
        # if lenth>=1000:
        #     lenth = 999
        # print(lenth)
        random_points = np.random.choice(lenth, 8, replace=False)

        F8 = eightpoint (pts1[random_points, :], pts2[random_points, :], M)
        # print(F8)
        pts1_h = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
        l2 = F8 @ pts1_h
        # print(l2)
        l2 = l2/np.sqrt(np.sum(l2[:2, :]**2, axis=0))


        pts2_h = np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))
        deviation = abs(np.sum(pts2_h*l2, axis=0))

        # determine the inliners
        inliers = np.transpose(deviation < tol)

        if inliers[inliers].shape[0] > maximum_inliers:
            maximum_inliers = inliers[inliers].shape[0]
            F = F8
            final_inliers = inliers

    print(maximum_inliers/pts1.shape[0])

    return F, final_inliers


'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
