import numpy as np
import NeuralNetwork as NN
import matplotlib.pyplot as plt
import pandas as pd


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
for x_val in np.linspace(0, 1, 200):
    for y_val in np.linspace(0, 1, 200):
        z = nn.predict([[x_val], [y_val]])
        points.append([x_val, y_val, z[0,0]])
points = np.array(points)
plt.scatter(points[:,0], points[:,1], c=points[:,2], cmap="viridis", s=10, alpha=0.5)
plt.show()


df_train = pd.read_csv(r"datasets\mnist_train.csv")
X_train = df_train.drop(columns=["label"]).values.astype(np.float32)
y_train = df_train["label"].values.astype(np.int32)

X_train = X_train / 255.0
X_train = X_train.reshape(-1, 784, 1)

# Normalize labels to [0,1] for sigmoid
Y_train = y_train / 9.0
Y_train = Y_train.reshape(-1, 1, 1)

df_test = pd.read_csv(r"datasets\mnist_test.csv")
X_test = df_test.drop(columns=["label"]).values.astype(np.float32)
y_test = df_test["label"].values.astype(np.int32)

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 784, 1)

Y_test = y_test / 9.0  # Normalize to [0,1]
Y_test = Y_test.reshape(-1, 1, 1)

nn2 = NN.NeuralNetwork(
    NN.Layer(784, 392),
    NN.Sigmoid(),
    NN.Layer(392, 191),
    NN.Sigmoid(),
    NN.Layer(191, 128),
    NN.Sigmoid(),
    NN.Layer(128, 64),
    NN.Sigmoid(),
    NN.Layer(64, 28),
    NN.Sigmoid(),
    NN.Layer(28, 10),
    NN.Sigmoid(),
    NN.Layer(10, 1),
    NN.Sigmoid()
    #on testing with tanh activation and -1 to 1 normaliztion my best accuracy even with 0.09 MSE was 46.23%
    #sigmoid activation 0 to 1 normalization far exceeds this and beat out tanh mse on first epoch alone with 0.075
    #sigmoid final mse  0.0268 accuracy 38.22
    #might have to implement cross entropy loss for a decent multiclass output
)

nn2.train(NN.mse, NN.mse_der, X_train, Y_train, epochs=10, learning_rate=0.01)

mse_total = 0
correct = 0
total = len(X_test)

for i in range(total):
    output = nn2.predict(X_test[i])
    mse_total += NN.mse(Y_test[i], output)
    pred_val = output[0,0]
    # Map normalized values back to 0-9
    pred_digit = int(round(pred_val * 9))
    
    true_digit = y_test[i]
    if pred_digit == true_digit:
        correct += 1

avg_mse = mse_total / total
accuracy = correct / total * 100

print(f"Test MSE: {avg_mse:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
