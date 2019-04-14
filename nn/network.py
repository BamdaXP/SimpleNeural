from settings import settings
import nn.layers as layers
import numpy as np

class Network():

    

    def __init__(self,dataset,cost_type="MSE"):
        '''
        cost type:
        - MSE
        - CEH
        '''
        self.dataset = dataset#Matrix
        self.structure = list()
        self.cost_recorder = list()
        self.structure.append(layers.Layer(self.dataset.data.shape[1]))
        self.cost_type = cost_type


    def append_activation_layer(self,type="Sigmoid"):
        layer = layers.Layer.layer_types()[type]()
        layer.node_count = self.structure[-1].node_count
        self.structure.append(layer)
        

    def append_linear_layer(self,output_num,learning_rate=settings.DEFAULT_LEARNING_RATE,
                            regularization_coefficient=settings.DEFAULT_REGULARIZATION_COEFFICIENT,
                            learning_rate_update_mode="static",learning_rate_update_param=tuple()):
        self.structure.append(layers.LinearLayer(
            self.structure[-1].node_count,output_num,
            learning_rate,regularization_coefficient,
            learning_rate_update_mode,learning_rate_update_param))



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
    

    
    def cost_function(self):
        if self.cost_type == "CEH":       
            # Matrix
            shifted_exp_result = np.clip(np.exp(self.final_result - np.amax(self.final_result,axis=1,keepdims=True)),0.0,1e10)#number safety
            softmax_result = np.clip(shifted_exp_result/shifted_exp_result.sum(axis=1,keepdims=True),1e-10,1.0)
            return -self.dataset.target*np.log(softmax_result)
        else:
            return np.square(
                self.final_result - self.dataset.target)#.sum()/self.dataset.data.shape[0]

    def delta_cost(self):
        if self.cost_type == "CEH":
            #par_L/par_c = y - y_hat
            return self.final_result - self.dataset.target
        else:
            #par_L/par_c = 2*(y - y_hat)
            return 2.0*(self.final_result-self.dataset.target)


    
    def train(self,iter_count=1):
        #Storing all the result of each layer
        layer_results = self.forward_propagation()
        #Using original data as the first input,then each next layer uses the result of last layer
        layer_input_set = [self.dataset.data]+layer_results

        self.final_result = layer_results[-1]

        cost_function = self.cost_function()
        delta_cost = self.delta_cost()

        #Back propagation :stepping backward
        for l in range(len(self.structure))[::-1]:
            layer = self.structure[l]
            layer.backward(layer_input_set[l], delta_cost,iter_count=iter_count)

        #Return cost
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
    


