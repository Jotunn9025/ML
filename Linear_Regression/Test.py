import numpy as np
from Linear_Regression import LinearRegression
import matplotlib.pyplot as plt

X=np.linspace(0,1000,1000)
print(X)

noise = np.random.normal(0, 0.2, size=X.shape)
X=(X-X.min())/(X.max()-X.min())
model=LinearRegression(learning_rate=1e-3,iterations=int(1e5))
Y=10*X+20
Y=Y+noise
model.fit(X,Y)

y_pred=model.predict(X)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X,Y,color="lightgray",s=1,label="Experimental")
plt.plot(X,y_pred,color="red",label="predicted")
plt.legend()
plt.show()