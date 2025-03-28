import numpy as np
import scipy.io
from nn import *
from collections import Counter

import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
# print(valid_x.shape)

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(valid_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hiddenlayer1')
initialize_weights(hidden_size, hidden_size, params, 'hiddenlayer2')
initialize_weights(hidden_size, valid_x.shape[1], params, 'output')

keys = [key for key in params.keys()]
for k in keys:
    params['m_'+k] = np.zeros(params[k].shape)

training_loss = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
    prev_loss = 0
    for xb,_ in batches:
        # forward
        y1 = forward(xb, params, 'layer1', relu)
        y2 = forward(y1, params, 'hiddenlayer1', relu)
        y3 = forward(y2, params, 'hiddenlayer2', relu)
        probs = forward(y3, params, 'output', sigmoid)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        # loss, acc = compute_loss_and_acc(yb, probs)
        loss = np.sum((xb - probs)**2)
        # print(loss)
        total_loss = total_loss + loss
        delta1 = -2 * (xb-probs)
        # delta1 = prev_loss-loss
        prev_loss = loss
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'hiddenlayer2', relu_deriv)
        delta4 = backwards(delta3, params, 'hiddenlayer1', relu_deriv)
        backwards(delta4, params, 'layer1', relu_deriv)

    
        for k in params.keys():
            if '_' in k:
                continue
            params['m_'+k] = 0.9*params['m_'+k] - learning_rate * params['grad_'+k]
            params[k] += params['m_'+k]

    
    training_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9


X_axis = np.arange(max_iters)
plt.figure('loss')
plt.plot(X_axis, training_loss, color='b')
plt.legend('training')
plt.show()
        

y1 = forward(valid_x, params, 'layer1', relu)
y2 = forward(y1, params, 'hiddenlayer1', relu)
y3 = forward(y2, params, 'hiddenlayer2', relu)
valid_probs = forward(y3, params, 'output', sigmoid)

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################


for i in [5, 10, 100, 110, 500, 510, 1500, 1510, 2000, 2010]:
    plt.subplot(1,2,1)
    plt.imshow(valid_x[i].reshape(32,32).T)
    plt.subplot(1,2,2)
    plt.imshow(valid_probs[i].reshape(32,32).T)
    plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################

psnr = 0
for i in range(valid_probs.shape[0]):
    psnr = psnr + peak_signal_noise_ratio(valid_x[i], valid_probs[i])
psnr = psnr/valid_probs.shape[0]
print("PSNR", psnr)