import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################

    W = np.random.uniform(-2/(in_size+out_size), 2/(in_size+out_size), (in_size, out_size))
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    res = 1 / (1 + np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    ##### your code here #####
    ##########################

    pre_act = (X @ W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################

    # print("x",x)

    x_max = np.max(x, axis=1)
    # print("x_max",x_max)
    x_c = x - np.expand_dims(x_max, axis=1)
    # print("x_c", x_c)
    e_x = np.exp(x_c)
    # print("ex", e_x)
    e_x_sum = np.sum(e_x, axis=1)
    # print("sum", e_x_sum)

    res = e_x/np.expand_dims(e_x_sum, axis=1)
    # print("res", res)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################

    # print ("proobs", probs)
    loss = - np.sum( y * np.log(probs) ) / 100
    # print("loss", loss)

    y = y.astype(int)

    total = 0
    correct = 0
    for i in range (0, probs.shape[0]):
        class_pred = np.argmax(probs[i])
        class_truth = np.argmax(y[i])
        # print(class_pred, class_truth)
        if (class_pred == class_truth):
            correct = correct + 1
        total = total + 1
    
    acc = correct / (total+ 0.0000000001)

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################

    grad_W = np.zeros(W.shape)
    grad_X = np.zeros(X.shape)
    grad_b = np.zeros(b.shape)
    derivative = delta * activation_deriv(post_act)

    for i in range(0, X.shape[0]):
        grad_b = grad_b + derivative[i, :]
        grad_W = grad_W + (np.expand_dims(X[i, :], axis=1) @ np.expand_dims(derivative[i, :], axis=0))
        grad_X[i, :] = (W @ np.expand_dims(derivative[i, :], axis=1)).reshape([-1])

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################


    # print("x", x)
    # print("y", y)
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    shuffled_x = x[indices]
    shuffled_y = y[indices]

    num_batches = len(x) // batch_size
    shuffled_x = shuffled_x[:num_batches * batch_size]
    shuffled_y = shuffled_y[:num_batches * batch_size]



    # print("x", shuffled_x.shape)
    # print("y", shuffled_y.shape)

    # print(len(x))

    batches = []
    count = 0
    for i in range (0, num_batches):
        # print("x", np.array([shuffled_x[count:count+batch_size, :]]))
        # print("y", np.array([shuffled_y[count:count+batch_size, :]]))
        x_batch = shuffled_x[count:count+batch_size, :]
        y_batch = shuffled_y[count:count+batch_size, :]

        # print("x", x_batch.shape)
        # print("y", y_batch.shape)

        batch = [np.array(x_batch), np.array(y_batch)]
        # print(batch.shape)

        # batches.append(np.array([, ]))
        batches.append(batch)
        count = count+batch_size
         
    # batches = np.array(batches)


    
    # x_batches = np.reshape(shuffled_x, (x.shape[1], batch_size))
    # y_batches = np.reshape(shuffled_y, batch_size,)

    # y_batches = np.array(y_batches)
    # x_batches = np.array(x_batches)
    # print("x", x_batches.shape)
    # print("y", y_batches.shape)

    # batches = np.array([x_batches, y_batches])


    # arr = np.arange(len(x))
    # np.random.shuffle(arr)
    # print("arr", arr)
    # for i in range (0, len(x)):
    #     for 


    return batches
