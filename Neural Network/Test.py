import numpy as np
import NeuralNetwork as NN
import matplotlib.pyplot as plt
X = np.array([[0,0], [0,1], [1,0], [1,1]]).reshape(4,2,1)
Y = np.array([[0], [1], [1], [0]]).reshape(4,1,1)

nn = NN.NeuralNetwork(
    NN.Layer(2, 3),  
    NN.tanh(),
    NN.Layer(3, 1),  
    NN.tanh()
)

nn.train(NN.mse, NN.mse_der, X, Y, epochs=1000, learning_rate=0.1)

for x in X:
    pred = nn.predict(x)
    print(f"Input: {x.T} -> Prediction: {pred}")
    
points = []
for x in np.linspace(0, 1, 200):
    for y in np.linspace(0, 1, 200):
        z = nn.predict([[x], [y]])
        points.append([x, y, z[0,0]])
points = np.array(points)
plt.scatter(points[:,0],points[:,1],c=points[:,2],cmap="viridis",s=10,alpha=0.5)
plt.show()


