import numpy as np
from Linear_Regression import LinearRegression

X=np.arange(1,101,1)
print(X)
Y=X*20

model=LinearRegression(learning_rate=1e-6,iterations=int(1e4))

model.fit(X,Y)

print(model.predict([10,100,1000,5]))