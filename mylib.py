import pandas as pd
import pandas.core
import chainer.optimizers
import chainer.links as L
import chainer.functions as F
from chainer import Sequential


def data_column_conversion(data:pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    """
    Change the label element from str to int.

    @param data: the extracted data
    @return: the conversed data
    """
    data = data.assign(W = (data.label == 'W') + 0,D = (data.label == 'D') + 0,L = (data.label == 'L') + 0)
    data = data.drop("label",axis=1)
    return data


def data_randomization(data:pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    """
    Randomize data.

    @param data: the conversed data you want to separate
    @return: the randomized data
    """
    return data.sample(n = len(data))


def data_separate(data:pandas.core.frame.DataFrame, split_num:int) -> list:
    """
    Separate data.
    
    @param data: the data you want to separate
    @param split_num: the division number for k-fold
    @return: separated data list
    """
    data_separate = []
    for i in range(split_num):
        data_separate.append(data[i::split_num])
    return data_separate


def data_list_wdl_merge(data_list1:list, data_list2:list) -> list:
    """
    Join two data lists by array number.
    
    @param data_list1: the data list you want to join
    @param data_list2: the data list you want to join
    @return: Joined data list
    """
    list_size = len(data_list1)
    merged_data_list = []
    for i in range(list_size):
        merged_data_list.append(pd.concat([data_list1[i],data_list2[i]]))
    return merged_data_list


def assign_group_numbers_to_data(data_list:list) -> list:
    """
    Assign group numbers to data.
    Group number is array numbers of data list.
    
    @param data_list: the data list you want to assign group numbers
    @return data_list
    """
    list_size = len(data_list)
    for i in range(list_size):
        data_list[i] = data_list[i].assign(separate_num=i)
    return data_list


def data_list_put_together(data_list:list) -> pandas.core.frame.DataFrame:
    """
    Combine the data list into one DataFrame.
    
    @param data_list: the data list you want to assign group numbers
    @return collected data
    """
    list_size = len(data_list)
    data = data_list[0]
    for i in range(1,list_size):
        data = data.append(data_list[i])
    return data


def making_dataset_list_train(data:pandas.core.frame.DataFrame, split_num:int) -> list:
    """
    Create training data set list for cross validation using group number.
    
    @param data: the collected data
    @param split_num: the division number for k-fold
    @return training data set list
    """
    train_data_list = []
    for i in range(split_num):
        train_data_list.append(data[data['separate_num'] != i])
    for i in range(split_num):
        train_data_list[i] = train_data_list[i].drop(['separate_num'], axis = 1)
    return train_data_list


def making_dataset_list_val(data:pandas.core.frame.DataFrame, split_num:int) -> list:
    """
    Create validation data set list for cross validation using group number.
    
    @param data: the collected data
    @param split_num: the division number for k-fold
    @return validation data set list
    """
    val_data_list = []
    for i in range(split_num):
        val_data_list.append(data[data['separate_num'] == i])
    for i in range(split_num):
        val_data_list[i] = val_data_list[i].drop(['separate_num'], axis = 1)
    return val_data_list


def making_dataset_list_x(data_list:list) -> list:
    """
    Create data set list for only explanatory variables.
    
    @param data_list: data set list
    @return: data set list for only explanatory variables
    """
    list_size = len(data_list)
    for i in range(list_size):
        data_list[i] = data_list[i].drop(["W","D","L"],axis=1)
    return data_list


def making_dataset_list_y(data_list:list) -> list:
    """
    Create data set list for only target variables.
    
    @param data_list: data set list
    @return: data set list for only target variables
    """
    list_size = len(data_list)
    data_list_y = []
    for i in range(list_size):
        data_list_y.append(data_list[i][["W","D","L"]])
    return data_list_y


def translate_pandas_to_numpy(data_list:list) -> list:
    """
    Convert pandas to numpy and make type 'float32'.
    
    @param data_list:pandas data set list
    @return: numpy data set list
    """
    list_size = len(data_list)
    for i in range(list_size):
        data_list[i] = data_list[i].values.astype('float32')
    return data_list


def data_processing_for_stratified_sampling(data:pandas.core.frame.DataFrame, split_num:int) -> pandas.core.frame.DataFrame:
    """
    Process the data for stratified sampling.
    
    @param data: inputted data
    @param split_num: the division number for k-fold
    @return: the data for stratified sampling
    """
    win_data = data[data.label == 'W'] # Extract only the data of label of win
    draw_data = data[data.label == 'D'] # Extract only the data of label of draw
    lose_data = data[data.label == 'L'] # Extract only the data of label of lose
    data_list = [win_data, draw_data, lose_data]
    for i,wdl_data in enumerate(data_list):
        wdl_data = data_column_conversion(wdl_data)
        wdl_data = data_randomization(wdl_data)
        if i == 0:
            separated_data_list = data_separate(wdl_data, split_num)
        else:
            separated_data_list = data_list_wdl_merge(separated_data_list, data_separate(wdl_data, split_num))
    separated_data_list = assign_group_numbers_to_data(separated_data_list)
    integrated_data = data_list_put_together(separated_data_list)
    integrated_data = data_randomization(integrated_data)
    return integrated_data


def making_x_train_data_list_for_kfold(data:pandas.core.frame.DataFrame, split_num:int) -> list:
    """
    Making x train data list for k-fold cross validation.
    
    @param data: the data for stratified sampling
    @param split_num: the division number for k-fold
    @return: x train data set list
    """
    train_data_list = making_dataset_list_train(data, split_num)
    x_train_data_list = making_dataset_list_x(train_data_list)
    return translate_pandas_to_numpy(x_train_data_list)


def making_x_val_data_list_for_kfold(data:pandas.core.frame.DataFrame, split_num:int) -> list:
    """
    Making x validation data list for k-fold cross validation.
    
    @param data: the data for stratified sampling
    @param split_num: the division number for k-fold
    @return: x validation data set list
    """
    val_data_list = making_dataset_list_val(data, split_num)
    x_val_data_list = making_dataset_list_x(val_data_list)
    return translate_pandas_to_numpy(x_val_data_list)


def making_y_train_data_list_for_kfold(data:pandas.core.frame.DataFrame, split_num:int) -> list:
    """
    Making y train data list for k-fold cross validation.
    
    @param data: the data for stratified sampling
    @param split_num: the division number for k-fold
    @return: y train data set list
    """
    train_data_list = making_dataset_list_train(data, split_num)
    y_train_data_list = making_dataset_list_y(train_data_list)
    return translate_pandas_to_numpy(y_train_data_list)


def making_y_val_data_list_for_kfold(data:pandas.core.frame.DataFrame, split_num:int) -> list:
    """
    Making y validation data list for k-fold cross validation.
    
    @param data: the data for stratified sampling
    @param split_num: the division number for k-fold
    @return: y validation data set list
    """
    val_data_list = making_dataset_list_val(data, split_num)
    y_val_data_list = making_dataset_list_y(val_data_list)
    return translate_pandas_to_numpy(y_val_data_list)


def RPS(y_true, y_pred) -> float:
    """
    Calcurate loss by RPS.
    
    @param y_true: the answer list of target variable. ex.)(0.6,0.3,0.1)
    @param y_pred: the predict list of target variable. ex.)(0.4,0.4,0.2)
    @return: the value of loss
    """
    output = 0.
    data_num = len(y_true)
    for i in range(data_num):
        times = len(y_true[i]) - 1 
        cumulative_sum = 0.
        score = 0.
        for time in range(times):
            cumulative_sum += y_true[i,time] - y_pred[i,time]
            score += cumulative_sum ** 2
        score /= times
        output += score
    
    output /= data_num
    return output


def create_model_net(n_input,n_hidden,n_output):
    """
    Create net of model.
    
    @param n_input: number of neurons in input layer
    @param n_hidden: number of neurons in hidden layer
    @param n_hidden: number of neurons in output layer
    @return: net
    """
    net = Sequential(
        L.Linear(n_input, n_hidden), F.relu,
        L.Linear(n_hidden, n_hidden), F.relu,
        L.Linear(n_hidden, n_output), F.softmax)
    return net


def create_model_optimizer(net,alpha):
    """
    Create optimizer of model.
    
    @param net: net of using model
    @param alpha: learning rate
    @return: optimizer
    """
    optimizer = chainer.optimizers.Adam(alpha=alpha)
    optimizer.setup(net)
    return optimizer