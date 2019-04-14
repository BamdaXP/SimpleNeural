import numpy as np
import nn.network as nt

from util import ploter
from util.evaluator import Evaluator
from dataset.dataset_fetcher import Dataset, Sineset
if __name__ == "__main__":

    train_set = Dataset()
    test_set = Dataset(type="test")
    

    print(train_set.target)
    print(train_set.data.shape)

    sin_set = Sineset()
    sin_test = Sineset(type="test")


    network = nt.Network(train_set,cost_type="CEH")

    # Hiden layer
    network.append_linear_layer(64)
    network.append_activation_layer(type="ReLU")
    network.append_linear_layer(32)
    network.append_activation_layer(type="ReLU")
    network.append_linear_layer(16)
    network.append_activation_layer(type="ReLU")
    network.append_linear_layer(8)
    network.append_activation_layer(type="ReLU")
    network.append_linear_layer(4)

    # network.show_structure()
    network.train_repeatly(times=1000,print_cost=True)


    #print(network.final_result)
    ploter.plot_cost(network)

    self_evaluator = Evaluator(network,network.dataset)
    evaluator = Evaluator(network,test_set)
    self_evaluator.clf_evaluate()
    evaluator.clf_evaluate()
    print(train_set.target)
    print(self_evaluator.result)

