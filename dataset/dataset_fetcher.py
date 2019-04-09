from settings import settings
import numpy as np
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

class Dataset():
    '''
    A wrapper for training dataset fetched by sklearn.

    ## Vectorized data shape:
    -----------------------------      
    obj1 |  feature1     |feature2    |...    |feature_n   |
    obj2 |
    .... |
    obj_n|
    -----------------------------

    ## bunch : Bunch object with the following attribute:

    - bunch.data: list, length [n_samples]
    - bunch.target: array, shape [n_samples]
    - bunch.filenames: list, length [n_samples]
    - bunch.DESCR: a description of the dataset.
    - bunch.target_names: a list of categories of the returned data,
      length [n_classes]. This depends on the `categories` parameter.
    '''
    def __init__(self,type="train"):
        
        if  type == "test":
            self.__bunch = fetch_20newsgroups(
                subset='test',  categories=settings.CATEGORIES)
        else:
            self.__bunch = fetch_20newsgroups(
                subset='train',  categories=settings.CATEGORIES)
        #The vectorizer to vectorize text features
        self.__vectorizer = TfidfVectorizer(max_features=settings.MAX_FEATURES)

        
        self.__data_raw = self.__bunch.data#Raw data
        # Data vectorized in to a feature matrix
        self.__data = self.__vectorizer.fit_transform(
            self.__data_raw).toarray()
        self.__target = self.__bunch.target



    #Fetched data is archieved and can not be modified
    #Using property getter to avoid modification
    @property
    def bunch(self):
        return self.__bunch
    @property
    def data_raw(self):
        return self.__data_raw
    @property
    def data(self):
        return self.__data
    @property
    def target(self):
        '''
        Returning a matrix with (sample_num,category_count) shape
        For example:
        [0,1,3...]
        =>
        [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
        ]
        '''
        t = np.zeros(shape=self.__target.shape+(len(settings.CATEGORIES),))
        for i in range(self.__target.shape[0]):
            t[i,self.__target[i]] = 1
        return t



    def show(self):
        pprint("Showing testing dataset:\n    Data type:%s\n    Vectorized data:%s \n    Target Vector:%s \nEnd showing testing dataset. \n\n" 
            % (self.data.dtype, self.data,self.target))



class Sineset():
    def __init__(self,type="train"):
        if type == "test":
            self.data = np.linspace(np.pi*0.7, np.pi, 60).reshape(60,1)
        else:
            self.data = np.linspace(-np.pi, 0.7 * np.pi, 140).reshape(140,1)
        
        self.target = np.sin(self.data)
