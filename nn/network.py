from settings import settings
import nn.layers as layers
import numpy as np

class Network():

    @staticmethod
    def layer_types():
        return {
            None:layers.Layer,
            "Sigmoid":layers.SigmoidLayer,
            "ReLU":layers.ReLULayer,
            "Tanh":layers.TanhLayer,
        }

    def __init__(self,dataset):
        self.dataset = dataset#Matrix
        self.structure = list()
        self.cost_recorder = list()

    def append_activation_layer(self,type="Sigmoid"):
        self.structure.append(Network.layer_types()[type]())

    def append_linear_layer(self,i_num,o_num,learning_rate=settings.DEFAULT_LEARNING_RATE,regularization_coefficient=settings.DEFAULT_REGULARIZATION_COEFFICIENT,learning_rate_update_mode="static",learning_rate_update_param=tuple()):
        self.structure.append(layers.LinearLayer(i_num,o_num,learning_rate,regularization_coefficient,learning_rate_update_mode,learning_rate_update_param))

    def forward_propagation(self,input_data=None):
        layer_results = []

        if isinstance(input_data,np.ndarray):
            layer_input = input_data
        else:
            layer_input = self.dataset.data

        for layer in self.structure:
            layer_results.append(layer.forward(layer_input))
            layer_input = layer_results[-1]

        return layer_results
    
    def train(self,iter_count=1):
        #Storing all the result of each layer
        layer_results = self.forward_propagation()
        #Using original data as the first input,then each next layer uses the result of last layer
        layer_input_set = [self.dataset.data]+layer_results

        self.final_result = layer_results[-1]

        # Using default sum(y-y_hat)**2 as the cost function
        cost_function = np.square(self.final_result - self.dataset.target).sum()
        delta_cost = 2.0*(self.final_result-self.dataset.target)

        #Back propagation :stepping backward
        for l in range(len(self.structure))[::-1]:
            layer = self.structure[l]
            delta_cost = layer.backward(layer_input_set[l], delta_cost,iter_count=iter_count)

        return np.mean(cost_function)

    def train_repeatly(self, times, print_cost = False , print_interval=100, clear_record=True):
        if clear_record:
            self.cost_recorder.clear()

        for i in range(times):
            cost = self.train(i)
            self.cost_recorder.append(cost)
            if print_cost and i%print_interval == 0:
                print("Current cost:%s"%(cost))

    def predict(self,input_data):
        prediction = self.forward_propagation(input_data)[-1]
        return prediction


    #Helper function to show the network structure
    def show_structure(self):
        print("Showing network structure:")
        for l in self.structure:
            if type(l) == layers.LinearLayer:

                print(type(l).__name__+"\n=>")
                #print("\tWeights:\n%s\n\n"%(l.weights))
                #print("\tBias:\n%s\n=>\n"%(l.biases))
            else:
                print(type(l).__name__+"\n=>")
    


