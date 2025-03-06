import numpy as np

class LinearRegression:
    def __init__(self, *args, **kwargs):
        self.learning_rate = kwargs.get("learning_rate",0.01)
        self.iterations = kwargs.get("iterations",1000)
        self.seed=12
        self.W=None
        
        X=args[0]
        Y=args[1]
        
        try:
            self.fit(X,Y)
        except Exception as e:
            print(e.with_traceback())
        
    def fit(self,X,Y):
        X_i = np.c[np.ones((X.shape[0],1)),X] #Adds intercept to X so that line is more accurate
        np.random.seed(self.seed)
        self.W=np.random.randn(X_i.shape[1])
        
        for iteration in range(self.iterations):
            y_pred = X_i.dot(self.W)
            errors = y_pred - Y
            
            gradients = (2/len(X_i)) * X_i.T.dot(errors)
            
            self.W = self.W - self.learning_rate * gradients
            
        return self
    
    def predict(self,X):
        X = np.x_[np.ones(X.shape[0],1),X]
        return X.dot(self.W)