import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 150
# pick a batch size, learning rate
batch_size = 40
learning_rate = 0.01
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

# fig = plt.figure()
# imgrid = ImageGrid(fig, 111,nrows_ncols=(8, 8), )

# for i in range(hidden_size):
#     imgrid[i].imshow(np.reshape(params['Wlayer1'][:, i], (32, 32))) 
#     plt.axis('off')

# plt.show()

training_accuracy = []
test_accuracy = []
training_loss = []
test_loss = []

predictions = []
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
                # forward
        y1 = forward(xb, params, 'layer1')
        probs = forward(y1, params, 'output', softmax)

        # predictions.append(probs)
        # print(np.array(predictions).shape)
        # print(predictions)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)
        # apply gradient
        params['Wlayer1'] = params['Wlayer1'] - (learning_rate * params['grad_Wlayer1'])
        params['blayer1'] = params['blayer1'] - (learning_rate * params['grad_blayer1'])
        params['Woutput'] = params['Woutput'] - (learning_rate * params['grad_Woutput'])
        params['boutput'] = params['boutput'] - (learning_rate * params['grad_boutput'])

    total_acc /= batch_num
    training_accuracy.append(total_acc)
    training_loss.append(total_loss)

    
    y1 = forward(valid_x, params, 'layer1')
    probs = forward(y1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    test_loss.append(loss)
    test_accuracy.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))


X_axis = np.arange(max_iters)
plt.figure('accuracy')
plt.plot(X_axis, training_accuracy, color='b')
plt.plot(X_axis, test_accuracy, color='r')
plt.legend(['training', 'validation'])
plt.show()

plt.figure('loss')
plt.plot(X_axis, training_loss, color='b')
plt.plot(X_axis, test_loss, color='r')
plt.legend(['training', 'validation'])
plt.show()

# run on validation set and report accuracy! should be above 75%
valid_acc = None
valid_y = valid_y.astype(int)
y1 = forward(valid_x, params, 'layer1')
probs_valid = forward(y1, params, 'output', softmax)
predictions = probs_valid
loss, valid_acc = compute_loss_and_acc(valid_y, probs_valid)

##########################
##### your code here #####
##########################

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################
fig = plt.figure()
imgrid = ImageGrid(fig, 111,nrows_ncols=(8, 8), )

for i in range(hidden_size):
    imgrid[i].imshow(np.reshape(params['Wlayer1'][:, i], (32, 32))) 
    plt.axis('off')

plt.show()


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# predictions = np.array(predictions)
# print("train_y",valid_y.shape)
# print("pred", probs_valid.shape)

total = 0
correct = 0
# confusion_matrix = np.zeros(shape=(8,8))
for i in range(0, valid_y.shape[0]):
    # if(i*10==0):
    # print(i)
    # print("acc", (correct/total))
    # print("valid_y", valid_y[i])
    # print("pred", predictions[i])
    truth = np.argmax(valid_y[i])
    guess = np.argmax(predictions[i])
    confusion_matrix[truth][guess] = confusion_matrix[truth][guess]+1
    total = total + 1
    if (truth == guess):
        correct = correct +1
acc = correct/total
print ("confusion accuracy ",correct/total)
print(confusion_matrix)

# compute comfusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()