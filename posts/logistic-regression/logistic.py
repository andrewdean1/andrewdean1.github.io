import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # return the dot product of X and w: gives a vector of scores with one entry
        # per data point
        return X @ self.w

        

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        
        scores = self.score(X)
        return (scores > 0.5).float()

class LogisticRegression(LinearModel):

    def __init__(self, w):
        self.w = w

    # Calculates the sigmoid for a given score s
    def sigmoid(self, s):
        '''
        Takes in a score s 
        Returns the sigmoid function for given s
        '''
        return 1 / (1 + torch.exp(-s))

    # Calculates the empirical risk
    def loss(self, X, y):
        '''
        Takes in data set X and target y
        Returns empirical loss
        '''
        s = self.score(X)
        sigma = self.sigmoid(s)

        first_half = -sigma.log() * y
        second_half = -(1 - sigma).log() * (1 - y)

        return (1 / X.size(0)) * (first_half + second_half).sum()

    # Gradient of empirical risk
    def grad(self, X, y):
        '''
        Takes in data set X and a target y
        Returns gradient of empirical risk
        '''
        s = self.score(X)
        sigma = self.sigmoid(s)
        sigma = sigma[:, None]

        y = y[:, None]

        grad = (sigma - y) * X

        return grad.mean(dim=0)


class GradientDescentOptimizer:
    def __init__(self, model, w, w_prev):
        self.model = model 
        self.w = w
        self.w_prev = w_prev

    # One step of the logistic regression update
    def step(self, X, y, alpha, beta):
        '''
        Takes in feature matrix (X), target vector (y), learning rate (alpha), and momentum (beta)
        Then updates w and w_prev based on parameters
        '''
        temp = self.model.w - alpha * (self.model.grad(X, y)) + beta * (self.model.w - self.w_prev)
        self.w_prev = self.model.w
        self.model.w = temp