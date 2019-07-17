import pandas as pd
import numpy as np
import chainer.optimizers
import mylib

# constant setting
split_num = 10
n_input = 8
n_hidden = 20
n_output = 3
alpha = 0.0001
n_epoch = 50
batchsize = 16


# input data 
path = 'trainrat_new.txt'
data = pd.read_csv(path,sep=' ')

path2 = 'testrat_new.txt'
data2 = pd.read_csv(path2,sep=' ')

data = pd.concat([data,data2])


"""
This cell outputs the predicted value for test data.
A model is built from scratch to train with all data.
"""

# create optimizer
production_net = mylib.create_model_net(n_input,n_hidden,n_output)
production_optimizer = mylib.create_model_optimizer(production_net,alpha)

# make dataset
c_data = mylib.data_column_conversion(data)
c_data = mylib.data_randomization(c_data)
x_train = c_data.drop(["W","D","L"],axis=1)
x_train = x_train.values.astype('float32')
y_train = c_data[["W","D","L"]]
y_train = y_train.values.astype('float32')


# log storage
results_train = []
iteration = 0
for epoch in range(n_epoch):
    loss_list = []
    for i in range(0, len(x_train), batchsize):
        # batch preparation
        x_train_batch = x_train[i:i+batchsize,:]
        y_train_batch = y_train[i:i+batchsize,:]

        # output predicted value
        y_train_batch_pred = production_net(x_train_batch)
        
        # apply objective function to calculate classification accuracy
        loss_train_batch = mylib.RPS(y_train_batch, y_train_batch_pred)
        loss_list.append(loss_train_batch.array)

        # slope reset and slope calculation
        production_net.cleargrads()
        loss_train_batch.backward()

        # parameter update
        production_optimizer.update()

        # count up
        iteration += 1
        
    # output objective function for training data, and aggregate classification accuracy
    loss_train = np.mean(loss_list)
    
    # display the result
    print('epoch: {}, iteration: {}, loss (train): {:.4f}'.format(epoch, iteration, loss_train))
    
    # log storage
    results_train.append(loss_train)
    

# input data
path3 = 'predrat_20190708.txt'
data3 = pd.read_csv(path3,sep=',')

# making test dataset
test_data = data3.drop(["num","label"],axis=1)
test_data = test_data.values.astype('float32')

# output predicted value
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    test_pred = production_net(test_data)

# unit dataset and predicted value
output_data = np.concatenate([test_data, test_pred.data], 1)

# convert to pandas
columns = ['HHATT','HHDEF','HAATT','HADEF','AHATT','AHDEF','AAATT','AADEF','W','D','L']
output = pd.DataFrame(data=output_data, columns=columns, dtype='float64')

# output file
output.to_csv("output.txt", sep=",")
