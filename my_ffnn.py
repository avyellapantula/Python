import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import matplotlib

## dataset
# mean and std for the x belonging to the first class
mu_x1, sigma_x1 = 0, 0.1

# Constant to make the sec distrib diff from the first
# x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5
x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0, 1, 0, 1

# creating the first distribution
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 0,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 0,
                   'type': 0})

d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 0,
                   'type': 1})

d3 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 0,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'type': 0})

d4 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'type': 1})

data = pd.concat([d1, d2, d3, d4], ignore_index=True)

## dataset created

class FFNN(object):
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        """output size is our bias, and we're giving it val 1"""
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size

        self.o_error = 0
        self.o_delta = 0
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.z2_error = 0

        # the whole weight matrix is here
        # till the hidden layer
        # weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        # final set of weights from the hidden layer till the output layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self,X):
        """Forward propogation through our network"""
        X['bias'] = 1 # adding 1 to the inputs to include the bias in the weight
        self.z1 = np.dot(X,self.w1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = sigmoid(self.z1) # activation func
        self.z3 = np.dot(self.z2, self.w2) # dot product of hidden layer (z2) and second set
        # of 3x1 weights
        o = sigmoid(self.z3) # final activation function
        return o
    def predict(self, X):
        return self.forward(self, X)
    """
    we need to implement back prop of the error
    to adjust the weights and reduce the error
    use the backward() method
    we start from the output and calculate the error (of residuals)
    this is used to calculate the delta
    which is used to update the weights
    in all layers we take the output of the neurons and use it as input
    passing it through the sigmoid_prime func
    and multiplying it by the error and the step (learning Rate)
    """
    def backward(self, X, y, output, step):
        """Backward prop of the errors"""
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.o_error = y - output  # error in output
        self.o_delta = self.o_error * sigmoid_prime(output) * step  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(
            self.w2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * sigmoid_prime(self.z2) * step  # applying derivative of sigmoid to z2 error

        self.w1 += X.T.dot(self.z2_delta)  # adjusting first of weights
        self.w2 += self.z2.T.dot(self.o_delta)  # adjusting second set of weights

    """
    fit() method
    training the model for each data pt
    doing two passes, one fwd, one backwd
    """
    def fit(self, X, y, epochs=10, step=0.05):
        for epoch in range(epochs):
            X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
            output = self.forward(X)
            self.backward(X, y, output, step)

# external functions
# we're going to be using the sigmoid funcs as activation funcs

def sigmoid(s):
    """Activation function"""
    return 1/(1+np.exp(-s))
def sigmoid_prime(s):
    """Derivative of the sigmoid"""
    return sigmoid(s) * (1-sigmoid(s))

# our ffnn is ready and it can be used for our task
# splitting the dataset in training and test set
msk = np.random.rand(len(data)) < 0.8

# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1', 'x2']][msk], data[['type']][msk].values

# everything else goes into the validation set
test_x, test_y = data[['x1','x2']][~msk], data[['type']][~msk].values

# training the network
my_network = FFNN()
my_network.fit(train_x,train_y,epochs=10000,step=0.001)

# verifying the performance of the algo
pred_y = test_x.apply(my_network.forward, axis=1)

# reshaping the data
test_y_ = [i[0] for i in test_y]
pred_y_ = [i[0] for i in pred_y]

print('MSE: ', mean_squared_error(test_y, pred_y_))
print('AUC: ', roc_auc_score(test_y, pred_y))

"""
The MSE, after 1,000 epochs is less than 0.01—a pretty good result. We measured the performance by using the ROC
which measures how good we were to order our predictions. 
With an AUC of over 0.99, we are confident that there are a few mistakes, but the model is still working very well. 
It's also possible to verify the performances using a confusion matrix. In this case, we will have to fix a threshold to discriminate between predicting one label or another. 

As the results are separated by a large gap, a threshold of 0.5 seems appropriate:"""

threshold = 0.5
pred_y_binary = [0 if i > threshold else 1 for i in pred_y_]
cm = confusion_matrix(test_y_, pred_y_binary, labels=[0, 1])

print(pd.DataFrame(cm,index=['True 0', 'True 1'],columns=['Predicted 0', 'Predicted 1']))