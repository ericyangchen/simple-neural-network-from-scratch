
# DLP Lab 1

## Introduction
In lab 1, we implement a simple neural network consisting of an input layer, two hidden layers, and an output layer.\
The neural network is trained using the backpropagation algorithm, and the Mean Squared Error (MSE) is used as the loss function.\
The neural network is implemented using only the NumPy library.

## Experiment setup

### Sigmoid function
The sigmoid function is used as the activation function in the hidden layer of the neural network. The sigmoid function is defined as follows:
```
sigmoid(x) = 1 / (1 + np.exp(-x))
```
There are some properties of the sigmoid function:
- The sigmoid function is differentiable.
- The sigmoid function is monotonically increasing.
- The sigmoid function is bounded between 0 and 1.

### Loss function
The Mean Squared Error (MSE) is used as the loss function. The MSE is defined as follows:
```
MSE = 1/N * sum((y - y_pred)^2)
```

### Neural network
The neural network is implemented with the following structure:
- Input layer: 2 neurons
- 1st Hidden layer: 4 neurons
- 2nd Hidden layer: 4 neurons
- Output layer: 1 neuron

<img src="assets/experiment-setup/neural-network.png" width="500">


### Backpropagation
The backpropagation algorithm is used to train the neural network. The backpropagation algorithm is implemented as follows:
1. Forward pass
    - Calculate the output of each layer.
2. Backward pass
    - Calculate the gradient of the loss function with respect to the output of each layer.
    - Update the weights of each layer using the gradient descent algorithm.

<img src="assets/experiment-setup/backpropagation.png" width="700">


## Results

### Linear dataset

**parameters:**
- epochs: 100000
- learning rate: 0.1
- hidden unit size: 16


**Training loss/accuracy:**\
<img src="assets/result/training_loss_and_accuracy_linear_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="500">

**Ground truth & Prediction:**\
<img src="assets/result/result_linear_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">

**Learning Curve / Accuracy Curve:**\
<img src="assets/result/learning_curve_and_accuracy_curve_linear_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">

**Testing Accuracy:**\
<img src="assets/result/testing_accuracy_linear_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="500">


### XOR dataset

**parameters:**
- epochs: 100000
- learning rate: 0.1


**Training loss/accuracy:**\
<img src="assets/result/training_loss_and_accuracy_xor_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="500">

**Ground truth & Prediction:**\
<img src="assets/result/result_xor_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">

**Learning Curve:**\
<img src="assets/result/learning_curve_and_accuracy_curve_xor_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">

**Testing Accuracy:**\
<img src="assets/result/testing_accuracy_xor_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="500">


## Discussion  


### Different learning rate
**Training linear dataset with learning rate 0.1, 0.01, 0.001:**
<img src="assets/different_lr/learning_curve_and_accuracy_curve_linear_FullyConnectedNeuralNetwork_sigmoid_100000_16.png" width="1000">

We can see that the learning rate affects the convergence speed of the neural network. A higher learning rate leads to faster convergence, but it may also lead to overshooting.
\
**Training XOR dataset with learning rate 0.1, 0.01, 0.001:**
<img src="assets/different_lr/learning_curve_and_accuracy_curve_xor_FullyConnectedNeuralNetwork_sigmoid_500000_16.png" width="1000">

For the XOR dataset, the 0.01 and 0.001 learning rates are too small, and the neural network cannot converge until 500,000 or doesn't converge at all. The 0.1 learning rate achieves the best performance.


### Different number of hidden units
**Training linear dataset with number of hidden units 16, 32, 64:**
<img src="assets/different_hidden_size/learning_curve_and_accuracy_curve_linear_FullyConnectedNeuralNetwork_sigmoid_30000_16.png" width="1000">

We can see that the number of hidden units affects the capacity of the neural network. A higher number of hidden units leads to higher capacity, which allows the network to learn more complex patterns. However, a higher number of hidden units may also leads to overfitting. But in this case, there's no overfitting, as we have the same training/testing dataset. Hence, the higher number of hidden units converges faster.


**Training XOR dataset with number of hidden units 16, 32, 64:**
<img src="assets/different_hidden_size/learning_curve_and_accuracy_curve_xor_FullyConnectedNeuralNetwork_sigmoid_100000_16.png" width="1000">

We can see that for hidden units = 16, the network converges much slower than hidden units = 32, 64. The network with hidden units = 64 converges the fastest.

### Without activation function
**Training linear dataset without activation function:**
<img src="assets/without_activation/learning_curve_and_accuracy_curve_linear_FullyConnectedNeuralNetwork_none_100000_0.1_16.png" width="1000">
<img src="assets/without_activation/result_linear_FullyConnectedNeuralNetwork_none_100000_0.1_16.png" width="1000">

We can see that without the activation function, the neural network is still able to classify the linear dataset pretty well.

**Training XOR dataset without activation function:**
<img src="assets/without_activation/learning_curve_and_accuracy_curve_xor_FullyConnectedNeuralNetwork_none_100000_0.1_16.png" width="1000">
<img src="assets/without_activation/result_xor_FullyConnectedNeuralNetwork_none_100000_0.1_16.png" width="1000">

We can see that without the activation function, the neural network is unable to classify the XOR dataset.


## Extra

### Implement different optimizers
I implemented the Momentum optimizer.

**Training linear dataset with Momentum optimizers:**
The learning curve shows that the momentum updates the weights faster than the normal gradient descent. The accuracy curve shows that the momentum optimizer converges faster than the normal gradient descent.
<img src="assets/momentum_optimizer/learning_curve_and_accuracy_curve_linear_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">

<img src="assets/momentum_optimizer/result_linear_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">


**Training XOR dataset with Momentum optimizers:**
Same for the XOR dataset, the momentum optimizer converges faster than the normal gradient descent.
<img src="assets/momentum_optimizer/learning_curve_and_accuracy_curve_xor_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">
<img src="assets/momentum_optimizer/result_xor_FullyConnectedNeuralNetwork_sigmoid_100000_0.1_16.png" width="1000">



### Implement different activation functions
I implemented the `Sigmoid`, `ReLU`, and `tanh` activation functions. 
The model with ReLU activation function converges the fastest on linear dataset

**Training linear dataset with different activation functions:**
<img src="assets/different_activation/learning_curve_and_accuracy_curve_linear_FullyConnectedNeuralNetwork_sigmoid_100000_16.png" width="1000">

The different activation functions seem to not affect the model convergence speed too much on the linear dataset.
\
**Training XOR dataset with different activation functions:**
<img src="assets/different_activation/learning_curve_and_accuracy_curve_xor_FullyConnectedNeuralNetwork_sigmoid_100000_16.png" width="1000">

For the XOR dataset, the ReLU activation function converges fastest, while tanh being second, and sigmoid being the last.
