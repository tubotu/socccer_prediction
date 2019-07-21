"""
It is a library used in "main_cross_validation.py" and "main_predict_test.py".
It contains functions to create model.
"""
import chainer.optimizers
import chainer.links as L
import chainer.functions as F
from chainer import Sequential

def create_net(n_input,n_hidden,n_output):
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


def create_optimizer(net,alpha):
    """
    Create optimizer of model.
    
    @param net: net of using model
    @param alpha: learning rate
    @return: optimizer
    """
    optimizer = chainer.optimizers.Adam(alpha=alpha)
    optimizer.setup(net)
    return optimizer