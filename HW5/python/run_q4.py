import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    # print(bboxes)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # exit()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    threshold = (bboxes[0][2] - bboxes[0][0])/2
    # bboxes = sorted(bboxes, key=lambda b: b[0])
    bboxes.sort(key=lambda x: x[2])
    # print("tres", threshold)
    rows = []
    row = [bboxes[0]]
    for i in range(1, len(bboxes)):
        # print(abs(bboxes[i][2] -  bboxes[i-1][2]))
        if (  abs(bboxes[i][2] -  bboxes[i-1][2]) <= threshold):
            row.append(bboxes[i])
        else:
            row.sort(key=lambda x: x[1])
            rows.append(row)
            row = [bboxes[i]]
    rows.append(row)
    # print("rows", rows)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    data = []
    for row in rows:
        row_letters = []
        for y1, x1, y2, x2 in row:

            try:

                buffer = 20
                crop = bw[y1-buffer:y2+buffer, x1-buffer:x2+buffer]

                # skimage.io.imshow(crop)
                # plt.show()
                # exit()
                
                
                height = y2-y1
                width = x2-x1


                diff = abs(height - width)
                pad_left = pad_right = pad_top = pad_bottom = 0

                if height < width:
                    pad_top = pad_bottom = diff // 2
                    if diff % 2 != 0:
                        pad_bottom += 1
                elif width < height:
                    pad_left = pad_right = diff // 2
                    if diff % 2 != 0:
                        pad_right += 1

                # Pad the image to make it square with ones
                crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=1)

                # skimage.io.imshow(crop)
                # print("shape", crop.shape)
                # plt.show()
                # exit()


                crop = skimage.transform.resize(crop, (32, 32))
                crop = skimage.morphology.erosion(crop, kernel)
                crop = np.transpose(crop)
                row_letters.append(crop.flatten())
            except:
                print("Error")
        data.append(np.array(row_letters))
    ##########################
    ##### your code here #####
    ##########################
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
             30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
    for row_letters in data:
        y1 = forward(row_letters, params, 'layer1')
        probs = forward(y1, params, 'output', softmax)
        predicted_string = ''
        for i in range(probs.shape[0]):
            class_index = np.argmax(probs[i])
            predicted_string += classes[class_index]

        print(predicted_string)
    