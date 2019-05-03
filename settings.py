
class Settings():

    '''
    Settings for datafecther and vectorizer    
    '''
    # The text categories to fetch and classify
    CATEGORIES = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']
    MAX_FEATURES = 500  # Max features to vectorize

    '''
    Settings for ploter   
    '''
    PLOT_PADDING = 0.5
    GRID_SIZE = 0.01


    '''
    Settings for neural network  
    '''
    DEFAULT_LEARNING_RATE = 0.00005
    DEFAULT_REGULARIZATION_COEFFICIENT = 0.05


settings = Settings()
