import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

# initiating random number
np.random.seed(11)
# Creating the dataset
# mean and standard deviation for the x belonging to the first class
mu_x1, sigma_x1 = 0, 0.1
# constant to make the second distribution different from the first
x2_mu_diff = 0.35
# creating the first distribution
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000),'x2': np.random.normal(mu_x1, sigma_x1 , 1000),'type': 0})
# creating the second distribution
d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff, 'x2': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff, 'type': 1})
data = pd.concat([d1, d2], ignore_index=True)
# plotting
# ax = sns.scatterplot(x="x1", y="x2", hue="type",data=data)
# plt.show()

'''creating perceptron class'''
class Perceptron (object):
    """
    simple implementation of perceptron algorithm
    """

    def __init__(self, w0=1, w1=.1, w2=.1):
        # weights
        self.w0 = w0 # bias
        self.w1 = w1
        self.w2 = w2
    '''
    We now need to add the methods to calculate the prediction to our class, which refers to the
    part that implements the mathematical formula. Of course, at the beginning, we don't know
    what the weights are (that's actually why we train the model), but we need some values to
    start, so we initialize them to an arbitrary value.
    We will use the step function as our activation function for the artificial neuron, which will
    be the filter that decides whether the signal should pass
    '''

    def step_function(self,z):
        if z>=0:
            return 1
        else:
            return 0

    '''
    The input will be then summed and multiplied by the weights, so we will need to 
    implement a method that will take two pieces of input and return their weighted sum. 
    '''
    def weighted_sum_inputs(self, x1, x2):
        wsum = sum([1*self.w0,x1*self.w1,x2*self.w2])
        return wsum

    def predict(self, x1, x2):
        """
        Uses the step function to determine the output
        """
        z = self.weighted_sum_inputs(x1, x2)
        return self.step_function(z)

    def predict_boundary(self, x):
        """
        Used to predict the boundaries of our classifier
        """
        return -(self.w1 * x + self.w0) / self.w2

    def fit(self, X, y, epochs=1, step=0.1, verbose=True):
        """
        Train the model given the dataset
        """
        errors = []
        for epoch in range(epochs):
            error = 0
            for i in range(0, len(X.index)):
                x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
                # The update is proportional to the step size and
                # the error
                update = step * (target - self.predict(x1, x2))
                # this is multiplying learning rate by resids
                self.w1 += update * x1
                self.w2 += update * x2
                self.w0 += update
                # we see here that the weighted error is multiplied and
                # added to corresponding weight
                error += int(update != 0.0)
            errors.append(error)
            if verbose:
                print('Epochs: {} - Error: {} - Errors from all epochs:{}'
                      .format(epoch, error, errors))
        # we call this strat the Perceptron Learning Rule
        # if the problem is linearly separable, then the
        # perceptron Learning Rule will find a set of
        # weights that will solve the problem in a
        # finite set of iterations

    """
    we would like to be able to visualize these results
    on the input space, by drawing a linear decision boundary
    we added in the following method later
    """
    def predict_boundary(self, x):
        """
        used to predict the boundaries of our classifier
        """
        return -(self.w1*x + self.w0)/self.w2

'''Perceptron class complete'''


# Splitting the dataset in training and test set
msk = np.random.rand(len(data)) < 0.8
# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1','x2']][msk], data.type[msk]
# Everything else will go into the valitation set
test_x, test_y = data[['x1','x2']][~msk], data.type[~msk]

# initialize the weights
my_perceptron = Perceptron(0.1,0.1)
my_perceptron.fit(train_x,train_y,epochs=1,step=0.005)

# we must be able to check the algo's performance. how?
# we use the confusion matrix, will show all of the correct
# predictions and missclassifications

# note that its a binary task, (but?)
# we will have 3 possible results
pred_y = test_x.apply(lambda x: my_perceptron.predict(x.x1,x.x2), axis=1)

cm = confusion_matrix(test_y, pred_y, labels=[0,1])

# let's see a visual version of this
print(pd.DataFrame(cm,index=['True 0','True 1'],columns=['Predicted 0','Predicted 1']))

# using predict_boundary mthd
# Adds decision boundary line to the scatterplot
ax = sns.scatterplot(x="x1", y="x2", hue="type",
data=data[~msk])
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = my_perceptron.predict_boundary(x_vals)
ax.plot(x_vals, y_vals, '--', c="red")
# plt.show()
#un-comment to see the plot
'''Implementing Perceptron in Keras
there r two ways
1, easiest is to create a keras model by using Sequential API
limitations: its not straightforward to define models that may have multiple
diff I/O sources
'''

# we start by initalizing the sequential class
my_perceptron2 = keras.Sequential()
'''
add our input layer
specify dimensions as well as
a few other parameters
we will also add a 'Dense' layer
'''
input_layer =Dense(1, input_dim=2, activation='sigmoid',kernel_initializer='zero')
my_perceptron2.add(input_layer)
# this gives a lot of warnings for ppl who don't have GPU set up
''' time to compile model
keras doesn't supply the step function because its not differentiable
will not work with backpropogation
define custom functions using: keras.backend

using MSE instead for simplicity.
As a gradient descent strategy, we use SGD (stochastic GD)
We specify a learning rate, which is 0.01
'''
# opt = keras.optimizers.Adam(learning_rate=0.01)
my_perceptron2.compile(loss='mse', optimizer=SGD(lr=0.01))
# fit the model
my_perceptron2.fit(train_x.values, train_y, epochs=30, batch_size=1, shuffle=True)
# the batch size is the portion of the training set
# that will be used for one gradient iteration
# it will generally between 32 and 512 data pts
pred_y = my_perceptron2.predict(test_x)
print('MSE on the test set:', mean_squared_error(pred_y, test_y))


