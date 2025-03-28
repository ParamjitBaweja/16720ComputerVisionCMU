# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
import cv2
from matplotlib import pyplot as plt

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    B = None
    L = None

    u, s, v = np.linalg.svd(I, full_matrices=False)
    s[3:] = 0
    B = v[0:3, :]
    L = u[0:3, :]

    return B, L


if __name__ == "__main__":

    # Put your main code here
    # (b)
    I, L, s = loadData()
    B, L_new = estimatePseudonormalsUncalibrated(I)
    print("Original L", L)
    print("new L", L_new)
    


    miu = 0
    nu = 0
    labda = 5
    G = np.array([[1, 0, 0], [0, 1, 0], [miu, nu, labda]])
    B = np.linalg.inv(G.T).dot(B)



    albedos, normals=estimateAlbedosNormals(B)
    
    albedo_Image, normal_Image = displayAlbedosNormals(albedos, normals, s)
    # plt.imshow(albedo_Image, cmap='gray')
    # plt.show()
    # plt.imshow(normal_Image, cmap='rainbow')
    # plt.show()

    # cv2.imshow("albedo image", albedo_Image)
    # cv2.imshow("normal image", normal_Image)
    # cv2.waitKey(0)

    # (d)
    normals = enforceIntegrability(normals, s)
    surface = estimateShape(normals, s)
    min_val, max_val = np.min(surface), np.max(surface)
    surface_normalized = (surface - min_val) / (max_val - min_val)  
    surface_normalized = surface_normalized * 255 
    plotSurface(surface_normalized)
