import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import scipy
import time

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    start = time.time()



    M =  np.array(LucasKanadeAffine(image1, image2, threshold, num_iters))

    # M = np.array(InverseCompositionAffine(image1, image2, threshold, num_iters))

    end = time.time()
    print("this frame took: ",end - start)

    img_warp = scipy.ndimage.affine_transform( np.array(image1), -M)

    diff_images = abs (img_warp - image2)
    # print(diff_images.shape)
    # diff_images[diff_images<=tolerance] = 0
    # diff_images[diff_images>tolerance] = 1
    diff_images = np.where(diff_images > tolerance, 0, 255)/255.0
    mask = diff_images
    # print(mask[0])
    mask = scipy.ndimage.binary_erosion(mask).astype(mask.dtype)
    # print(mask[0])
    mask = scipy.ndimage.binary_dilation(mask).astype(mask.dtype)
    # print(mask[0])

    return mask
