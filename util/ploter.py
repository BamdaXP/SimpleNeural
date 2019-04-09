from settings import settings

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(hypo_func, vectorized_data):
    '''
    "hypo_func" is the pointer of hypothesis function.
    We only plot the first two features an show the boundary.
    '''
    # Set min and max values and give it some padding
    padding = settings.PLOT_PADDING

    x_min, x_max = vectorized_data[:, 0].min(
    ) - padding, vectorized_data[:, 0].max() + padding
    y_min, y_max = vectorized_data[:, 1].min(
    ) - padding, vectorized_data[:, 1].max() + padding

    h = settings.GRID_SIZE
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = hypo_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.ScalarMappable)
    plt.scatter(
        vectorized_data[:, 0], vectorized_data[:, 1], cmap=plt.cm.ScalarMappable)


#Wrappers for matplotlib
def plot_cost(network):
    plt.plot(network.cost_recorder)
    plt.show()


def plot_array(array):
    plt.plot(array)
    plt.show()
