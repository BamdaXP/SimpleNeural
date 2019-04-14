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


    network = nt.Network(sin_set,cost_type="MSE")

    # Hiden layer
    network.append_linear_layer(97)
    network.append_activation_layer(type="Tanh")
    network.append_linear_layer(1)

    # network.show_structure()
    network.train_repeatly(times=10000,print_cost=True)

    print(network.final_result)
    print(train_set.target)
    #print(network.final_result)
    ploter.plot_cost(network)

    self_evaluator = Evaluator(network,network.dataset)
    evaluator = Evaluator(network,sin_test)
    self_evaluator.reg_evaluate()
    evaluator.reg_evaluate()
