
class Settings():

    '''
    Settings for datafecther and vectorizer    
    '''
    # The text categories to fetch and classify
    CATEGORIES = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']
    MAX_FEATURES = 20  # Max features to vectorize

    '''
    Settings for ploter   
    '''
    PLOT_PADDING = 0.5
    GRID_SIZE = 0.01


    '''
    Settings for neural network  
    '''
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_REGULARIZATION_COEFFICIENT = 0.01
    DEFAULT_LEARNING_RATE_GAMMA = 1.0
    DEFAULT_LEARNING_RATE_STEP = 100


settings = Settings()
