# Machine Learning Repository

## Overview
This repository is a centralized collection of machine learning algorithms implemented from scratch. Each algorithm has its own subdirectory containing:

- The core implementation (e.g., as a Python class)
- A test script demonstrating usage
- A Jupyter Notebook explaining the implementation details and learnings

This is an ongoing project, and while I aim to manually implement each algorithm, I may not be able to do so for all.

## Structure
```
ML-Repository/
│── Linear_Regression/
│   ├── Linear_Regression.py   # Manually implemented Linear Regression as a class
│   ├── Test.py                # Script to test the implementation
│   ├── LinearRegression.ipynb # Jupyter Notebook explaining learnings
│── Neural_Networks/
│   ├── Neural_Network.py      # Manually implemented Neural Network class
│   ├── Test.py                # Script to test the implementation
│   ├── NeuralNetwork.ipynb    # Jupyter Notebook explaining learnings (yet to be added)
│── Other_Algorithm/           # Future algorithms will follow the same structure
│── README.md                  
```

## Running the Code
To run an algorithm's code, navigate to its subdirectory and execute the test script:
```sh
cd Linear_Regression
python Test.py
```

## Implemented Algorithms
### 1. Linear Regression
- Implemented manually using **Gradient Descent** (instead of the normal equation)
- Contains noise in generated data to simulate experimental errors
- Includes a Jupyter Notebook with detailed explanations

### 2. Neural Networks
- Implemented a basic **neural network** (not convolutional yet)
- Uses **Mean Squared Error (MSE)** loss function for regression problems and **Cross-Entropy** loss function for classification problems
- Works great for predicting **XOR**
- Supports **tanh, sigmoid, ReLU, SoftMax** activation functions
- Future additions include **Leaky ReLU, MISH** and a notebook explaining learnings

## Future Plans
- Add more machine learning algorithms (e.g., logistic regression, decision trees, convolutional neural networks)
- Maintain consistency in structure across implementations
- Improve efficiency and numerical stability where necessary

## Notes
- This repository is a **learning project**, not an authoritative source.
- Implementations may not be the most optimized but serve as an educational exercise.

Contributions and suggestions are welcome!

