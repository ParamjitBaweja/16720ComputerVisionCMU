'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import submission
import matplotlib.pyplot as plt
import helper

M = 640

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

F8 = submission.eightpoint(data['pts1'], data['pts2'], M)

intrinsics = np.load('../data/intrinsics.npz')
E = submission.essentialMatrix(F8, intrinsics['K1'], intrinsics['K2'])
np.savez('../data/q3_1.npz', E=E)
# print(E)

M2 = helper.camera2(E)
# print(M2[:,:,2])

# K2 = intrinsics['K2']
# print(K2.shape)
# print(M2[2].shape)

# C2 =  M2[2] @ K2 
# print(C2)



M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
C1 = intrinsics['K1'] @ M1

M2_final = []
P = []
for i in range(M2.shape[2]):
    temp = M2[:, :, i]
    C2 = intrinsics['K1'] @ temp
    W, error = submission.triangulate(C1, data['pts1'], C2, data['pts2'])

    if np.min(W[:, 2]) > 0:
        M2_final = temp
        P = W
        break

# print(M2_final.shape)
C2 = intrinsics['K2'] @ M2_final


np.savez('../data/q3_3.npz', M2=M2, C2=C2, P=P)

