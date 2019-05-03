# Simple Neural Network
A basic framework to construct a neural network system.

In this example,I used it to classify texts.(It is not well constructed)
## Usage
Creating a dataset:
```python
from dataset.dataset_fetcher import Dataset,Sinset
train_set = Dataset()#Classification dataset
sin_set = Sinset()#Regression dataset
```

Creating a network:
```python
from nn.network import Network
network=Network(train_set)
```

Adding layers to the network:
```python
network.append_linear_layer(node_count)
network.append_activation_layer(type="Sigmoid")#ReLU , Tanh , etc
......
network.append_linear_layer(output_node_count)
```

Training your network:
```python
network.train_repeatly(train_count)
```

Reading result:
```python
network.final_result
```

Network evaluation:
```python
from util.evaluator import Evaluator:
#using train set as the test set to self-evaluate
self_evaluator = Evaluator(train_set)
self_evaluator.clf_evaluate()#classification evaluating
#self_evaluator.reg_evaluate()#regression evaluating
```

## Dependencies
* Numpy
* Matplotlib
* Scikit learn(Used to fetch text training source)

## To do
* ~~Sine function is well predicted~~
* Text classifier did not work well,accuracy remains about 40%,suffering from overfitting problem.
