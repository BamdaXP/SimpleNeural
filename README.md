# Simple Neural Network
A basic framework to construct a neural network system.

In this example,I used it to classify texts.(It is not well constructed)
## Usage
Creating a dataset:
from dataset.dataset_fetcher import Dataset,Sinset
train_set = Dataset()#Classification dataset
sin_set = Sinset()#Regression dataset


Creating a network:
from nn.network import Network
network=Network(train_set)


Adding layers to the network
network.append_linear_layer(train_set.data.shape[0],node_count)
network.append_activation_layer()
......
network.append_linear_layer(last_layer_node_count,output_node_count)


Training your network:
network.train_repeatly(train_count)


Reading result:
network.final_result


Network evaluation:
from util.evaluator import Evaluator:
self_evaluator = Evaluator(train_set)#using train set as the test set to self-evaluate
self_evaluator.clf_evaluate()#classification evaluating
#self_evaluator.reg_evaluate()#regression evaluating


## Dependencies
* Numpy
* Matplotlib
* Scikit learn(Used to fetch text training source)

## To do
* ~~Sine function is well predicted~~
* Text classifier did not work well,accuracy remains about 30%(Considering reconstructing the network)
