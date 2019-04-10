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


    node_count = 100

    network = nt.Network(train_set)

    # Input layer
    network.append_linear_layer(
        network.dataset.data.shape[1], node_count)

    # Hiden layer
    network.append_activation_layer(type="Sigmoid")
    network.append_linear_layer(node_count, 4)

    # network.show_structure()
    network.train_repeatly(1000)


    print(network.final_result)
    ploter.plot_cost(network)

    self_evaluator = Evaluator(network,network.dataset)
    evaluator = Evaluator(network,test_set)
    self_evaluator.clf_evaluate()
    evaluator.clf_evaluate()

