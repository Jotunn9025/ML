import numpy as np

class LinearRegression:
    def __init__(self, *args, **kwargs):
        self.learning_rate = kwargs.get("learning_rate",0.01)
        self.iterations = kwargs.get("iterations",1000)
        self.seed=12
        self.W=None
        
        if len(args)>2:
            X=args[0]
            Y=args[1]
            try:
                self.fit(X,Y)
            except Exception as e:
                print(f"ERROR: {e}")
                
    def process_input(self, X, Y=None, fit_mode=True):
        X = np.asarray(X)
        
        if X.ndim == 0:
            X = X.reshape(1, 1)

        elif X.ndim == 1:
            if fit_mode:
                X = X.reshape(-1, 1)
            else:
                X = X.reshape(1, -1)
        
        if fit_mode:
            if Y is None:
                raise ValueError("Y cannot be None during fit.")
            Y = np.asarray(Y)
            if Y.ndim == 0:
                Y = np.array([Y])
            elif Y.ndim == 2 and Y.shape[1] == 1:
                Y = Y.flatten()
        
        return X, Y            
    
    def fit(self,X,Y):
        X_i = np.c_[np.ones((X.shape[0],1)),X] #Adds intercept to X so that line is more accurate
        np.random.seed(self.seed)
        self.W=np.random.randn(X_i.shape[1])
        
        for iteration in range(self.iterations):
            y_pred = X_i.dot(self.W)
            errors = y_pred - Y
            
            gradients = (2/len(X_i)) * X_i.T.dot(errors)
            
            NEW_W = self.W - self.learning_rate * gradients
            
            if np.any(np.isnan(NEW_W)) or np.any(np.isinf(NEW_W)):
                print(f"NaN or Inf encountered at iteration {iteration}")
                break
            self.W=NEW_W
        return self
    
    def predict(self,X):
        X=np.array(X)
        if X.ndim==0:
            X=np.array([[X]])
        elif X.ndim==1:
            X=X.reshape(-1,1)
        try:
            X = np.c_[np.ones((X.shape[0],1)),X]
            return X.dot(self.W)
        except Exception as e:
            print(f"ERROR: {e}")