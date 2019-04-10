import numpy as np
import nn.network as nt

from util import ploter
from util.evaluator import Evaluator
from dataset.dataset_fetcher import Dataset, Sineset
if __name__ == "__main__":

    train_set = Dataset()
    test_set = Dataset(type="test")

    sin_set = Sineset()
    sin_test = Sineset(type="test")

    node_cost = list()
    node_count = 100
    costs = []

    network = nt.Network(sin_set)
    # Input layer
    network.append_linear_layer(
        network.dataset.data.shape[1], node_count)

    # Hiden layer
    network.append_activation_layer(type="Sigmoid")
    #network.append_linear_layer(node_count, node_count)
    #network.append_activation_layer(type="ReLU")
    #network.append_linear_layer(node_count, node_count)
    #network.append_activation_layer(type="ReLU")
    network.append_linear_layer(node_count, 1)
    # Output layer

    # network.show_structure()
    network.train_repeatly(2000)
    costs.append(network.cost_recorder[-1])

    # node_cost.append(network.cost_recorder[-1])
    # ploter.plot_array(node_cost)

    #ploter.plot_array(costs)

    self_evaluator = Evaluator(network,network.dataset)
    evaluator = Evaluator(network,sin_test)
    self_evaluator.reg_evaluate()
    evaluator.reg_evaluate()

