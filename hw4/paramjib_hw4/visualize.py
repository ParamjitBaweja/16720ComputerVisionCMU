'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import submission
import matplotlib.pyplot as plt

M = 640
data1 = np.load('../data/some_corresp.npz')
F8 = submission.eightpoint(data1['pts1'], data1['pts2'], M)


data = np.load('../data/templeCoords.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
# print(data)
x1 = data['x1']
y1 = data['y1']
# print(x1.shape)
# print(np.dstack((x1[:,0], y1[:,0]))[0])
points1 = np.dstack((x1[:,0], y1[:,0]))[0]

points2 = []
for i in range(0,x1.shape[0]):

    points2.append(submission.epipolarCorrespondence(im1, im2, F8, x1[i, 0], y1[i, 0]))
points2= np.array(points2)

# print()
intrinsics = np.load('../data/intrinsics.npz')
M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
M2 = np.array([np.array([0.99942701,  0.03331428,  0.0059843,  -0.02601138]),
 np.array([ -0.03372743,  0.96531375,  0.25890503, -1.    ]),
 np.array([  0.00284851, -0.25895852,  0.96588424,  0.07981688])])
C1 = intrinsics['K1'] @ M1
C2 = intrinsics['K1'] @ M2
print(points2.shape)
points, err = submission.triangulate(C1,points1, C2, points2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('3D Reconstruction of Points')
plt.show()