import numpy as np
import math
from settings import settings
class Layer:
    def __init__(self):
        pass


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


class ReLULayer(Layer):
    def __init__(self):
        pass

    def forward(self, layer_input):
        return np.maximum(0, layer_input)

    def backward(self, layer_input, layer_delta_output, iter_count):
        relu_grad = layer_input > 0
        return layer_delta_output*relu_grad

class SigmoidLayer(Layer):
    def __init__(self):
        pass

    def __sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def forward(self, layer_input):
        
        return self.__sigmoid(layer_input)

    def backward(self, layer_input, layer_delta_output, iter_count):
        grad_sigmoid = self.__sigmoid(
            layer_input)*(1-self.__sigmoid(layer_input))
        return layer_delta_output*grad_sigmoid


class TanhLayer(Layer):
    def __init__(self):
        pass

    def __tanh(self, x):
        return np.tanh(x)

    def forward(self, layer_input):
        return self.__tanh(layer_input)

    def backward(self, layer_input, layer_delta_output, iter_count):
        grad_tanh = 1-(self.__tanh(layer_input))**2
        return layer_delta_output*grad_tanh


class LinearLayer(Layer):

    def __init__(self, input_num, output_num, 
                learning_rate=settings.DEFAULT_LEARNING_RATE, regularization_coefficient=settings.DEFAULT_REGULARIZATION_COEFFICIENT, 
                learning_rate_update_mode="static", learning_rate_update_param=tuple()):

        
        self.learning_rate = learning_rate
        self.learning_rate_update_mode = learning_rate_update_mode
        self.learning_rate_param = learning_rate_update_param

        self.regularization_coefficient = regularization_coefficient
        self.weights = np.random.randn(input_num,output_num)#np.zeros(shape=(input_num,output_num))#
        self.biases = np.random.randn(output_num)*learning_rate#np.zeros(shape=(output_num,))#

  
    def forward(self, layer_input):
        return np.dot(layer_input, self.weights)+self.biases

    def backward(self, layer_input, layer_delta_output,iter_count):
        delta_input = np.dot(layer_delta_output, self.weights.T)
        #delta_W = par_l/par_W = sum(X*delta_Y)/m
        delta_weights = np.dot(
            layer_input.T, layer_delta_output)/layer_input.shape[0]
        #Calculate average of each column
        #delta_b= par_L/par_b = sum(delta_Y)/m   (X=1)
        delta_biases = layer_delta_output.mean(axis=0)
        self.weights = self.weights*(1.0-self.learning_rate*self.regularization_coefficient /#Regularization
                                    layer_input.shape[0]) - self.learning_rate*delta_weights
        self.biases = self.biases - self.learning_rate*delta_biases
        
        self.update_learning_rate(iter_count)

        return delta_input

    def update_learning_rate(self,iter_count):
        if self.learning_rate_update_mode == "static":
            pass
        elif self.learning_rate_update_mode == "step":
            #gamma,step
            self.learning_rate = self.learning_rate*self.learning_rate_param[0]**(iter_count/self.learning_rate_param[1])
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
