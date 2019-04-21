import numpy as np
import util.ploter as ploter 
from nn.network import Network
from dataset.dataset_fetcher import Textset
class Evaluator():

    correct_count = 0
    accuracy = 0.0
    @property
    def total_count(self):
        return self.testset.data.shape[0]

    def __init__(self,network:Network,testset:Textset):
        self.network = network
        self.testset = testset

    def clf_evaluate(self,clear=True):
        if clear:
            self.correct_count = 0
            self.accuracy = 0.0

        self.result = self.network.predict(
            input_data=self.testset.data)


        for i in range(self.total_count):

            if np.argmax(self.testset.target[i]) == np.argmax(self.result[i]):
                self.correct_count += 1
        
        self.accuracy = float(self.correct_count)/self.total_count
        print("Tested network correctly predicted %d out of %d in total.\nAccuracy:%f\n" % (
            self.correct_count, self.total_count, self.accuracy))

    def reg_evaluate(self):
        self.result = self.network.predict(
            input_data=self.testset.data)


        ploter.plot_array(self.result)
        
