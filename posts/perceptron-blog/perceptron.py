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

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """

        y_ = 2*y - 1
        return torch.mean((y_ * self.score(X) <= 0).float())

    def grad(self, X, y):
        """
        Computes the vector to add to w. 

        ARGUMENTS: 
            X, torch.Tensor: the observation of the feature matrix used in the current update.

            y, torch.Tensor: the element of the target vector used in the current update.                
        """
        s = X@self.w
        return (s*y < 0)*X*y


class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """

        n = X.size()[0]
        i = torch.randint(n, size = (1,))

        x_i = X[[i],:]
        y_i = y[i]

        # Record the loss before change
        current_loss = self.model.loss(X, y)

        # Update our perceptron, adding result to w
        self.model.w += torch.reshape(self.model.grad(x_i, y_i),(self.model.w.size()[0],))
        
        # Record the loss after change
        new_loss = self.model.loss(X, y)

        return i, abs(current_loss - new_loss)