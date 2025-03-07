import numpy as np
from Linear_Regression import LinearRegression
import matplotlib.pyplot as plt

X=np.array([np.linspace(0,1,1000),
            np.linspace(0,1,1000),
            np.linspace(0,1,1000)]).T
"""
    Below is a test of predicting MultiVariable Linear Regression using
    the formula Y=10*X1+20*X2-5*X3+78
    Pls do not implementation of LinearRegression only accepts X in the
    form where each column is a different variable and row is a different sample
    
    for X=scalar it auto converts to [[X]]
    for X=1D array it converts into a column vecotr form
"""

X=(X-X.min())/(X.max()-X.min())
model=LinearRegression(learning_rate=1e-3,iterations=int(1e5))
Y=10*X[:,0]+20*X[:,1]-5*X[:,2]+78

model.fit(X,Y)
pre=np.array([[100,100,100],
              [10,10,10]])#each column is a diff variable
y_pred=model.predict(pre)
print(y_pred)

X=np.linspace(0,1,100)
Y=50.21*X-312
model.fit(X,Y)
noise=np.random.normal(0,1,size=X.shape)
Y=Y+noise
y_pred=model.predict(X)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X,Y,color="lightgray",s=5,label="Experimental")
plt.plot(X,y_pred,color="red",label="predicted")
plt.legend()
plt.show()