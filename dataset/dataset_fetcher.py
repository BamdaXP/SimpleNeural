
from settings import settings
import numpy as np
from pprint import pprint


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import dataset.mnist as mnist

#import mnist
class Dataset():
    def __init__(self,data:np.ndarray,target:np.ndarray):
        self.data = data
        self.target = target

    @staticmethod
    def onehot(raw,types):
        '''
        A helper function returning a matrix with (sample_num,category_count) shape
        For example:
        [0,1,3...]
        =>
        [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
        ]
        
        '''
        t = np.zeros(shape=raw.shape+(types,))
        for i in range(raw.shape[0]):
            t[i, raw[i]] = 1

        return t

    @property
    def length(self):
        return len(self.data)

    def show(self):
        print("Data:\n%s"%(self.data))
        print("Target:\n%s"%(self.target))


class Textset(Dataset):
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
        self.data_raw = self.__bunch.data#Raw data
        # Data vectorized in to a feature matrix
        d = self.__vectorizer.fit_transform(self.data_raw).toarray()
        t = Dataset.onehot(self.__bunch.target,len(settings.CATEGORIES))
        super().__init__(data=d,target=t)



    #Fetched data is archieved and can not be modified
    #Using property getter to avoid modification
    @property
    def bunch(self):
        return self.__bunch
    @property
    def vectorizer(self):
        return self.__vectorizer



class Sineset(Dataset):
    def __init__(self,type="train"):
        if type == "test":
            d = np.linspace(np.pi*0.7, np.pi, 60).reshape(60,1)
        else:
            d = np.linspace(-np.pi, 0.7 * np.pi, 140).reshape(140,1)
        
        t = np.sin(self.data)

        super().__init__(data=d,target=t)

class Mnistset(Dataset):
    def __init__(self,type="train"):
        if type == "test":
            images = mnist.test_images()
            labels = mnist.test_labels()
        else:
            images = mnist.train_images()
            labels = mnist.train_labels()
        
        n_test, w, h = images.shape
        d = images.reshape((n_test, w*h))
        t = Dataset.onehot(labels,10)
        
        super().__init__(data=d,target=t)


