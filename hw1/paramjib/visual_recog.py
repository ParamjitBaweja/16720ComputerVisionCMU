import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import visual_words
import scipy.ndimage


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    # print("tile")
    K = opts.K
    hist, bin_edges = np.histogram(wordmap, bins=K+1)
    # normalized_hist = hist / np.sum(hist)
    # plt.plot(normalized_hist)
    # plt.show()
    # return normalized_hist
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    #lin_wordmap = np.reshape(wordmap, (wordmap.shape[0]*wordmap.shape[1]))
    # M = round(wordmap.shape[0] // math.pow(2,L))
    # N = round(wordmap.shape[1] // math.pow(2,L))
    # tiles = [wordmap[x:x+M,y:y+N] for x in range(0,wordmap.shape[0],M) for y in range(0,wordmap.shape[1],N)]
    # # np.array(tiles)
    # print(len(tiles))
    # #numpy.array_split(ary, indices_or_sections, axis=0)
    histograms = []
    for l in range(L + 1):
        weight = 1
        if l==0 or l==1:
            weight = math.pow(2.0, (-L))
        else:
            weight = math.pow(2.0, (l-L-1))
        num_divisions = int(math.pow(2,l))
        M = round(wordmap.shape[0] // math.pow(2,l))
        N = round(wordmap.shape[1] // math.pow(2,l))
        layer_histograms = []
        for i in range(num_divisions):
            y_start = i * M
            y_end = (i + 1) * M
            for j in range(num_divisions):
                x_start = j * N
                x_end = (j + 1) * N
                layer_histograms.append(get_feature_from_wordmap(opts, wordmap[y_start:y_end,x_start:x_end]))  # multiply by weight
        # print(len(layer_histograms)
        layer_histograms = layer_histograms/np.sum(layer_histograms)
        layer_histograms = layer_histograms*weight
        histograms.extend(layer_histograms)
        # print(len(histograms))
    concatenated_histograms = np.concatenate(histograms)
    # concatenated_histograms = concatenated_histograms/np.sum(concatenated_histograms)
    # print(concatenated_histograms.shape)
    # plt.plot(concatenated_histograms)
    # plt.show()
    return concatenated_histograms
    
    # plt.imshow(tiles[3])
    # plt.show()
    # ----- TODO -----
    # pass
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    pass

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))


    features = []
    for i in range (0, len(train_files)):
    #for i in range (0, 1):
        if i%10==0:
            print(i)
        img_path = join(opts.data_dir, train_files[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255.0
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        features.append( get_feature_from_wordmap_SPM(opts, wordmap))
    
        ## example code snippet to save the learned system
    print("features", len(features))
    print("labels", len(train_labels))
    print("dictionary", len(dictionary))
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


    # ----- TODO -----
    pass



def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # distances = scipy.spatial.distance.cdist([word_hist], histograms, 'euclidean')

    minima = np.minimum(word_hist, histograms)
    # intersection = np.sum(minima, axis=1)
    intersection = np.true_divide(np.sum(minima, axis=1), np.sum(histograms, axis=1))
    #distances = 1-distances
    print(intersection)
    return 1-intersection

    # # ----- TODO -----
    # pass    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    # print("hello")
    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    # print("no")
    # print(trained_system)
    dictionary = trained_system['dictionary']
    # print(len(dictionary))
    # print("ugh")

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    # print(test_opts.K)
    test_opts.L = trained_system['SPM_layer_num']
    # print(test_opts.L)
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    # print("i")
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    # print("come on")

    #print(trained_system)
    # ----- TODO -----

    total = 0
    correct = 0
    confusion_matrix = np.zeros(shape=(8,8))
    for i in range(0, len(test_files)):
    #for i in range(0, 200):
        # if(i*10==0):
        print(i)
        img_path = join(opts.data_dir, test_files[i])
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255.0
        wordmap = visual_words.get_visual_words(test_opts, img, dictionary)
        features = get_feature_from_wordmap_SPM(test_opts, wordmap)
        dist = distance_to_set(features, trained_system['features'])
        # print(dist)
        index = np.argmin(dist)
        # print(index)
        total = total+1
        confusion_matrix[trained_system['labels'][index]][test_labels[i]] = confusion_matrix[trained_system['labels'][index]][test_labels[i]]+1
        if (trained_system['labels'][index] == test_labels[i]):
            correct = correct +1
    acc = correct/total
    print ("accuracy ",correct/total)
    print(confusion_matrix)

    # # print(features)
    # # print("dictionary", dictionary)
    # # print("feat", features)
    # dist = distance_to_set(features, trained_system['features'])
    # dist = np.delete(dist[0], 40)
    # dist = np.delete(dist, 201)
    # #dist.remove(40)
    # print(len(dist))
    # print("dist", dist)
    # print(np.argmin(dist))
    # print(np.min(dist))
    # print(dist[201])
    # print(trained_system['features'][39])
    # print(trained_system['labels'][439])
    return confusion_matrix, acc

