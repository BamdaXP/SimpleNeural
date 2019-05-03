import numpy as np
import math
from settings import settings
class Layer:
    node_count = 0

    def __init__(self,node_count=0):
        self.node_count = node_count
        

    #Retrun layer output
    def forward(self, layer_input):
        return layer_input
    '''
    Use delta output to get delta_input 
    Which means:
    - layer_input:The original input data to the layer which is X
    - layer_delta_output:The delta output value to calculate delta_X,which is dY

    Returns:
    dX : layer delta input
    '''

    def backward(self, layer_input, layer_delta_output, iter_count):
        return layer_delta_output

class ActivationLayer(Layer):
    def __init__(self, type="Sigmoid", dropout_param=None):
        self.type = type
        self.dropout_param = dropout_param

    def __feature_func(self,x):
        if self.type == "Sigmoid":
            return 1.0/(1+np.exp(-x))
        elif self.type == "Tanh":
            return np.tanh(x)
        elif self.type == "ReLU":
            return np.maximum(x, 0)
        else:
             return x


    def forward(self,layer_input):
        result = self.__feature_func(layer_input)

        #Dropout
        if not self.dropout_param is None:
            level = self.dropout_param[0]
            if level < 0. or level >= 1:
                raise Exception('Dropout level must be in interval [0, 1)')

            retain_level = 1. - level
            sample = np.random.binomial(n=1, p=retain_level, size=result.shape)
            result *= sample
            result /= retain_level
        return result


    def backward(self,layer_input, layer_delta_output, iter_count):
        if self.type == "Sigmoid":
            grad_sigmoid = self.__feature_func(
                layer_input)*(1-self.__feature_func(layer_input))
            return layer_delta_output*grad_sigmoid
        elif self.type == "Tanh":
            grad_tanh = 1-(self.__feature_func(layer_input))**2
            return layer_delta_output*grad_tanh
        elif self.type == "ReLU":
            relu_grad = np.array(layer_input > 0, dtype=np.float)
            return layer_delta_output*relu_grad
        else:
            return layer_delta_output

class LinearLayer(Layer):

    def __init__(self, input_num, output_num, 
                learning_rate=settings.DEFAULT_LEARNING_RATE, regularization_coefficient=settings.DEFAULT_REGULARIZATION_COEFFICIENT, 
                learning_rate_update_mode="static", learning_rate_update_param=tuple()):
        self.node_count = output_num
        
        self.learning_rate = learning_rate
        self.learning_rate_update_mode = learning_rate_update_mode
        self.learning_rate_param = learning_rate_update_param

        self.regularization_coefficient = regularization_coefficient

        self.weights = np.random.randn(
            input_num, output_num)/np.sqrt(input_num)
        self.biases = np.zeros((1,output_num))

  
    def forward(self, layer_input):
        return np.dot(layer_input, self.weights) + self.biases

    def backward(self, layer_input, layer_delta_output,iter_count):

        delta_weights = np.dot(
            layer_input.T, layer_delta_output)
        delta_bias = np.sum(layer_delta_output,axis=0,keepdims=True)
        self.weights = self.weights - self.learning_rate*delta_weights
        self.biases = self.biases - self.learning_rate*delta_bias

        self.update_learning_rate(iter_count)

        layer_delta_input = np.dot(layer_delta_output,self.weights.T)

        return layer_delta_input


    def update_learning_rate(self,iter_count):
        if self.learning_rate_update_mode == "static":
            pass
        elif self.learning_rate_update_mode == "step":
            #gamma,step
            self.learning_rate = self.learning_rate*self.learning_rate_param[0]**np.floor(iter_count/self.learning_rate_param[1])
        elif self.learning_rate_update_mode == "exp":
            #gamma
            self.learning_rate = self.learning_rate*self.learning_rate_param[0]**iter_count
        elif self.learning_rate_update_mode == "inv":
            #gamma, power
            self.learning_rate = self.learning_rate*(
                1 + self.learning_rate_param[0] * iter_count) ** (- self.learning_rate_param[1])
        elif self.learning_rate_update_mode == "multistep":
            # need to implement
            pass
        elif self.learning_rate_update_mode == "poly":
            #max_iter,power
            self.learning_rate = self.learning_rate * (1.0 - iter_count/self.learning_rate_param[0]) ** (self.learning_rate[1])
        elif self.learning_rate_update_mode == "sigmoid":
            #gamma step
           self.learning_rate = self.learning_rate * (1.0/(1.0 + math.exp(-self.learning_rate_param[0] * (iter_count - self.learning_rate_param[1]))))
        else:
            pass


