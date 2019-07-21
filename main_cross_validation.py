"""
Do cross validation.
Please note that the data is using train and test together.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# dataset making for stratified sampling and k-fold
s_data = mylib.data_processing_for_stratified_sampling(data,split_num)
x_train = mylib.making_x_train_data_list_for_kfold(s_data,split_num)
x_val = mylib.making_x_val_data_list_for_kfold(s_data,split_num)
y_train = mylib.making_y_train_data_list_for_kfold(s_data,split_num)
y_val = mylib.making_y_val_data_list_for_kfold(s_data,split_num)


# create optimizer
optimizer = []
net = []
for i in range(split_num):
    net.append(mylib.create_model_net(n_input,n_hidden,n_output))
    optimizer.append(mylib.create_model_optimizer(net[i],alpha))


# for log storage
results_train_data = []
results_valid_data = []

for data_num in range(len(x_train)):
    # for log storage
    results_train = []
    results_valid = []
    # counting
    iteration = 0
    for epoch in range(n_epoch):
        # output of objective function for each batch and storage of classification accuracy
        loss_list = []

        for i in range(0, len(x_train[data_num]), batchsize):
            # batch preparation
            x_train_batch = x_train[data_num][i:i+batchsize,:]
            y_train_batch = y_train[data_num][i:i+batchsize,:]

            # output predicted value
            y_train_batch_pred = net[data_num](x_train_batch)
            # apply objective function to calculate classification accuracy
            loss_train_batch = mylib.RPS(y_train_batch, y_train_batch_pred)
            loss_list.append(loss_train_batch.array)

            # slope reset and slope calculation
            net[data_num].cleargrads()
            loss_train_batch.backward()

            # parameter update
            optimizer[data_num].update()

            # count up
            iteration += 1

        # output objective function for training data, and aggregate classification accuracy
        loss_train = np.mean(loss_list)

        # evaluate with validation data every time the epoch is over
        # output predicted values in validation data
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y_val_pred = net[data_num](x_val[data_num])

        # apply objective function to calculate classification accuracy
        loss_val = mylib.RPS(y_val_pred, y_val[data_num])

        # display the result
        print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
            epoch, iteration, loss_train, loss_val.array))

        # log storage
        results_train.append(loss_train)
        results_valid.append(loss_val.array)

    # log storage
    results_train_data.append(results_train)
    results_valid_data.append(results_valid)
    

# calcurate average
results_train_data_all = []
results_valid_data_all = []
results_train_data_all = np.zeros(epoch+1)
results_valid_data_all = np.zeros(epoch+1)
for i in range(split_num):   
    results_train_data_all += results_train_data[i]
    results_valid_data_all += results_valid_data[i]
results_train_data_ave = results_train_data_all / split_num
results_valid_data_ave = results_valid_data_all / split_num

# output of objective function
plt.plot(results_train_data_ave, label='train')  # set legend with label
plt.plot(results_valid_data_ave, label='valid')  # set legend with label
plt.legend()  # display legend
