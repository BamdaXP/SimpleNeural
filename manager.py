import numpy as np
import nn.network as nt

from util import ploter
from util.evaluator import Evaluator
from dataset.dataset_fetcher import Textset, Sineset,Mnistset

def text_clf():
    train_set = Textset()
    test_set = Textset(type="test")
    
    print(train_set.target)
    print(train_set.data.shape)

    network = nt.Network(train_set,cost_type="MSE")

    # Hiden layer
    network.append_linear_layer(16)
    network.append_activation_layer(type="ReLU")
    network.append_linear_layer(8)
    network.append_activation_layer(type="ReLU")
    network.append_linear_layer(4)

    # network.show_structure()
    network.train_repeatly(times=1000  ,print_cost=True)



    print(network.final_result)
    print(train_set.target)
    #print(network.final_result)
    ploter.plot_cost(network)

    self_evaluator = Evaluator(network,network.dataset)
    evaluator = Evaluator(network,test_set)
    self_evaluator.clf_evaluate()
    evaluator.clf_evaluate()


def sin_reg():
    sin_set = Sineset()
    sin_test = Sineset(type="test")

    network = nt.Network(sin_set, cost_type="MSE")

    # Hiden layer
    network.append_linear_layer(97)
    network.append_activation_layer(type="Tanh")
    network.append_linear_layer(1)

    # network.show_structure()
    network.train_repeatly(times=10000, print_cost=True)

    ploter.plot_cost(network)
    
    self_evaluator = Evaluator(network,network.dataset)
    evaluator = Evaluator(network,sin_test)
    self_evaluator.reg_evaluate()
    evaluator.reg_evaluate()


def mnist_clf():
    print("OK")
    mnist_set = Mnistset(type="train",selection_range=(0,3000))

    print("OK")

    print(mnist_set.data.shape)
    print(mnist_set.target)

    network = nt.Network(mnist_set,cost_type="MSE")
    network.append_linear_layer(16,learning_rate=0.0000001)
    network.append_activation_layer(type="Sigmoid")
    network.append_linear_layer(8,learning_rate=0.0000001)
    network.append_activation_layer(type="Sigmoid")
    network.append_linear_layer(4,learning_rate=0.0000001)
    network.append_activation_layer(type="Sigmoid")
    network.append_linear_layer(10,learning_rate=0.0000001)

    network.train_repeatly(100,print_cost=True,print_interval=1)

    network.dataset.selection_range = (3000,6000)
    self_evaulator = Evaluator(network,network.dataset)
    self_evaulator.clf_evaluate()
    

if __name__ == "__main__":
    mnist_clf()
