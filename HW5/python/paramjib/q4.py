import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

from matplotlib import pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    # print("before", image.shape)
    image = skimage.filters.gaussian(image)
    # skimage.io.imshow(image)
    # plt.show()
    # print("after", image.shape)
    # print("hello")

    image = skimage.restoration.denoise_bilateral(image, channel_axis=-1)
    # skimage.io.imshow(image)
    # plt.show()

    grayscale = skimage.color.rgb2gray(image)

    threshold = skimage.filters.threshold_otsu(grayscale)
    bw = grayscale < threshold
    morph = skimage.morphology.closing(bw, skimage.morphology.square(5))


    labels = skimage.morphology.label(morph, connectivity=2)
    letters = skimage.measure.regionprops(labels)

    area = sum([i.area for i in letters])/len(letters)
    bboxes = [i.bbox for i in letters if i.area > area/3]


    # skimage.io.imshow(np.invert(bw))
    # plt.show()
    # print("bw", bw.shape)
    # print("bw")

    return bboxes, np.invert(bw)