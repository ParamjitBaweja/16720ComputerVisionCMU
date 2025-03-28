import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
from sklearn.cluster import KMeans
import util

import matplotlib.pyplot as plt

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    H = img.shape[0]
    W = img.shape[1]
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    filter_scales = opts.filter_scales
    img = skimage.color.rgb2lab(img)
    # img = (img+110)/220
    responses = np.zeros(shape= (H,W,4*3*len(filter_scales)))
    #responses = np.empty(shape= (H,W,4*3*len(filter_scales)))

    for i in range(0, len(filter_scales)):
        gauss = np.zeros(shape=(H,W,3))
        gauss_lap = np.zeros(shape=(H,W,3))
        gauss_x = np.zeros(shape=(H,W,3))
        gauss_y = np.zeros(shape=(H,W,3))
        for j in range(0,3):

            gauss[:,:,j] = scipy.ndimage.gaussian_filter(img[:,:,j], filter_scales[i])
            gauss_lap[:,:,j] = scipy.ndimage.gaussian_laplace(img[:,:,j], filter_scales[i])
            gauss_x[:,:,j] = scipy.ndimage.gaussian_filter(img[:,:,j], filter_scales[i], order = [1,0])
            gauss_y[:,:,j] = scipy.ndimage.gaussian_filter(img[:,:,j], filter_scales[i], order = [0,1])

            #print(np.shape(gauss_y))
            # print(gauss_lap.shape())
            # print(gauss_x.shape())
            # print(gauss_y.shape())
            #print(np.dstack((gauss, gauss_lap,gauss_x, gauss_y)).shape)
            # print(np.shape(np.stack((gauss, gauss_lap,gauss_x, gauss_y), axis = 2)))

        responses[:,:,(12*i)+0:(12*i)+3] = gauss  
        responses[:,:,(12*i)+3:(12*i)+6] = gauss_lap 
        responses[:,:,(12*i)+6:(12*i)+9] = gauss_x 
        responses[:,:,(12*i)+9:(12*i)+12] = gauss_y
        #responses[:,:,i:i+12] = np.stack((gauss, gauss_lap,gauss_x, gauss_y), axis = 2)
        # print("bleh", gauss.shape)
        # temp = np.stack((gauss, gauss_lap,gauss_x, gauss_y), axis = 2)

        # np.append(responses, temp)
        # print(responses.shape)

    # print(opts)
    # plt.imshow(lap_gauss    )
    # plt.show()
    # plt.imshow(gauss)
    # plt.show()
    # print(filter_scales)
    # ----- TODO -----

    return responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # num_images = 1
    # opts.alpha = 25
    num_images = len(train_files)
    filter_responses = np.zeros(shape=( (opts.alpha*num_images) , 4*3*len(opts.filter_scales)))
    for i in range (0, num_images):
        img_path = join(opts.data_dir, train_files[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255.0
        responses = extract_filter_responses(opts, img)
        if i%10 ==0:
            print(i)
        H = img.shape[0] - 1
        W = img.shape[1] - 1
        np.random.seed(15)
        H_rand = np.random.uniform(0,H,opts.alpha)
        W_rand = np.random.uniform(0,W,opts.alpha)
        # print("shapeH", H_rand.shape)

        for j in range (0, opts.alpha):

            # H_rand = round(H_rand[i])
            # W_rand = round(np.random.uniform(0,W,1)[0])
            filter_responses[(opts.alpha*i)+j, : ] = responses[round(H_rand[j]), round(W_rand[j]), :]
    # print(filter_responses)
    # filter_responses = np.reshape(filter_responses, (50,50,36))
    # util.display_filter_responses(opts, filter_responses)
    # read all images
    # extract alpha T filter responses over training files
    # call k means


    # print("starting")
    kmeans = sklearn.cluster.KMeans(n_clusters=K, verbose =1 ).fit(filter_responses)
    print("kmeans")
    dictionary = kmeans.cluster_centers_
    print("dictionary")
    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    # print(dictionary)
    responses = extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, responses)

    # print("responses shape", responses.shape)
    # print("shape of dictionary", dictionary.shape)
    wordmap = np.zeros(shape=(img.shape[0], img.shape[1]))
    responses = np.reshape(responses, (responses.shape[0]*responses.shape[1], responses.shape[2]))
    # print("responses at 10", responses[10])
    # print("responses at 10000", responses[10000])

    distances = scipy.spatial.distance.cdist(responses, dictionary, 'euclidean')
    # print(distances.shape)
    # print(distances[10])
    # print("distance for the next pixel")
    # print(distances[10000])
    wordmap = distances.argmin( axis=1)
    # print(wordmap[10])
    wordmap = np.reshape(wordmap, (img.shape[0],img.shape[1]))


    # for i in range(0, img.shape[0]):
    #     for j in range(0,img.shape[1]):
    #         distances = scipy.spatial.distance.cdist([responses[i,j,:]], dictionary, 'euclidean')
    #         # print("distance array ",distances)
    #         wordmap[i,j]=np.argmin(distances,axis=1)
            # print(wordmap[i,j])
            # print(np.min(distances))
            # print("point")
            # print(distances[0])
            # print(distances.shape)
            # print("argmin", np.argmin(distances[0,:]))
            # [temp] = np.where(distances[0,:] == np.min(distances))[0]
            # print(temp)
            # wordmap[i][j] = distances[0,temp]
            # img [i][j] = 
    # ----- TODO -----
    return wordmap

