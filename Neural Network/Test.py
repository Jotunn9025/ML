import numpy as np
import NeuralNetwork as NN
import matplotlib.pyplot as plt
import pandas as pd


# X = np.array([[0,0], [0,1], [1,0], [1,1]]).reshape(4,2,1)
# Y = np.array([[0], [1], [1], [0]]).reshape(4,1,1)

# nn = NN.NeuralNetwork(
#     NN.Layer(2, 3),
#     NN.tanh(),
#     NN.Layer(3, 1),
#     NN.tanh()
# )

# nn.train(NN.mse, NN.mse_der, X, Y, epochs=1000, learning_rate=0.1)

# for x in X:
#     pred = nn.predict(x)
#     print(f"Input: {x.T} -> Prediction: {pred}")

# points = []
# for x_val in np.linspace(0, 1, 200):
#     for y_val in np.linspace(0, 1, 200):
#         z = nn.predict([[x_val], [y_val]])
#         points.append([x_val, y_val, z[0,0]])
# points = np.array(points)
# plt.scatter(points[:,0], points[:,1], c=points[:,2], cmap="viridis", s=10, alpha=0.5)
# plt.show()


df_train = pd.read_csv(r"datasets\mnist_train.csv")
X_train = df_train.drop(columns=["label"]).values.astype(np.float32)
y_train = df_train["label"].values.astype(np.int32)

df_test = pd.read_csv(r"datasets\mnist_test.csv")
X_test = df_test.drop(columns=["label"]).values.astype(np.float32)
y_test = df_test["label"].values.astype(np.int32)

def one_hot_encode(labels, num_classes):
    return [[1 if i == label else 0 for i in range(num_classes)] for label in labels]

Y_train = np.array(one_hot_encode(y_train, 10)).reshape(-1,10,1)
Y_test = np.array(one_hot_encode(y_test, 10)).reshape(-1,10,1)

X_train = X_train.reshape(-1, 784, 1)
X_test = X_test.reshape(-1, 784, 1)
X_test,Y_test,X_train,Y_train=X_test/255,Y_test,X_train/255,Y_train

nn2 = NN.NeuralNetwork(
    NN.Layer(784, 150),
    NN.tanh(),
    NN.Layer(150, 28),
    NN.tanh(),
    NN.Layer(28, 10), 
    NN.Softmax(one_true=True)
    #94.24% 
)


nn2.train_with_accuracy(NN.cross_entropy, NN.cross_entropy_der, X_train, Y_train,X_test,Y_test, epochs=10, learning_rate=0.01)

error_total = 0
correct = 0
total = len(X_test)

for i in range(total):
    output = nn2.predict(X_test[i])
    error_total += NN.cross_entropy(Y_test[i], output)
    pred_digit = int(np.argmax(output))
    true_digit = int(np.argmax(Y_test[i]))
    if pred_digit == true_digit:
        correct += 1

avg_error = error_total / total
accuracy = correct / total * 100

print(f"Test Accuracy: {accuracy:.2f}%")
