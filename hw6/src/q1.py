# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import cv2
import skimage.color
from matplotlib import cm

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    image = None
    
    x_grid, y_grid = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    x_coords = pxSize * (x_grid - (res[0] / 2)) + center[0]
    y_coords = pxSize * (y_grid - (res[1] / 2)) + center[1]

    z = rad ** 2 - x_coords ** 2 - y_coords ** 2

    mask = z < 0
    z[mask] = 0
    z = np.sqrt(z)

    points = np.stack((x_coords, y_coords, z), axis=2).reshape((res[0] * res[1], -1))
    normalized_points = (points.T / np.linalg.norm(points, axis=1).T).T

    image = np.dot(normalized_points, light).reshape((res[1], res[0]))
    image[mask] = 0

    return image


def loadData(path = "../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    images = []
    for i in range(0,7):
        image_path = path + "input_" + str(i+1) + ".tif"
        Im = cv2.imread(image_path, -1)
        Im_xyz = skimage.color.rgb2xyz(Im)
        Im_y = Im_xyz[:, :, 1].reshape(-1, 1)
        images.append(Im_y)
    
    I = np.asarray(images).reshape(len(images), -1)
    L = np.load(path + 'sources.npy').T
    sample_image = cv2.imread(image_path, -1)
    s = sample_image.shape[:2]
    return I, L, s




def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    B = np.linalg.inv((L @ L.T)) @ L @ I
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None

    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos 
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    albedoIm = np.reshape(albedos, s)

    normals = normals + abs(np.min(normals))
    normals = normals / np.max(normals)

    colorShape = np.array(s)
    colorShape = np.append (colorShape, 3)
    normalIm = np.reshape(normals.T, colorShape)

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    z = normals[0,:] / (-normals[2,:])
    y = normals[1,:] / (-normals[2,:])
    zx = np.reshape(z, s)
    zy = np.reshape(y, s)
    surface = integrateFrankot(zx, zy)

    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    h, w = surface.shape
    x = np.arange(w)
    y = np.arange(h)
    X_grid, Y_grid = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, surface, edgecolor='none', cmap=cm.coolwarm)
    ax.set_title('Surface plot')
    plt.show()

    pass


if __name__ == '__main__':

    # Put your main code here
    # (b)
    center = [0, 0, 0]
    rad = 0.75
    pxSize = 7e-4
    res = [3840, 2160]
    lights = [([1, 1, 1]/np.sqrt(3)), ([1, -1, 1] /np.sqrt(3)), ([-1, -1, 1]/np.sqrt(3))]
    if False:
        for i in range(len(lights)):
            print(lights[i])
            image = renderNDotLSphere(center, rad, lights[i], pxSize, res)
            # plt.imshow(image)
            # plt.show()
            cv2.imshow("image", image)
            cv2.waitKey(0)

    # (c)
    I, L, s = loadData()
    # print(I, L, s)

    # (d)
    u, v, vh = np.linalg.svd(I, full_matrices=False)
    print(v)

    # (e)
    B = estimatePseudonormalsCalibrated(I,L)
    albedos, normals=estimateAlbedosNormals(B)
    
    # (f)
    albedo_Image, normal_Image =displayAlbedosNormals(albedos, normals, s)
    plt.imshow(albedo_Image, cmap='gray')
    plt.show()
    plt.imshow(normal_Image, cmap='rainbow')
    plt.show()
    # cv2.imshow("albedo image", albedo_Image)
    # cv2.imshow("normal image", normal_Image)
    # cv2.waitKey(0)

    # (i)
    surface = estimateShape(normals, s)
    min_val, max_val = np.min(surface), np.max(surface)
    surface_normalized = (surface - min_val) / (max_val - min_val)   
    plotSurface(surface_normalized)


    pass
