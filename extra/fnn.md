# Backpropagation Basics

1. **Forward pass**:

Input data is passed through the network to generate predictions. Each neuron in the network applies weights to the inputs, performs computations (a weighted sum followed by a non-linear activation function), and produces an output. This output is then used as input to the next layer in the network

2. **Backward pass (backpropagation = computing gradient)**: 

The cost function, often the squared difference between the predicted and actual output (error), is calculated. The backpropagation algorithm is then used to propagate this error backward through the network. This is done by calculating the derivative of the error with respect to each weight in the network. This derivative effectively measures how much each weight contributed to the error. In other words, it gives us the rate of change of the error with respect to a small change in the weight. For each neuron, the weights are adjusted based on their contribution to the error:

- If a weight has a large impact on the error, a small adjustment is made because small change in these weights have large impact on activation of target neuron
- If a weight has a small impact on the error, a larger adjustment is made because large change in these weights have smaller impact on activation of target neuron

The backpropagation algorithm computes the gradients, which are vectors with the same dimension as the weights. The values in these vectors represent the rate of change of the error with respect to a change in the corresponding weight.

- If the gradient is positive, the weight is decreased to minimize the cost. The size of the decrease is proportional to the value of the gradient.
- If the gradient is negative, the weight is increased to minimize the cost. The size of the increase is also proportional to the value of the gradient.

This process is repeated for each layer in the network, starting from the output layer and moving backward to the input layer. Hence, the term "backpropagation". The goal is to adjust the weights in a way that minimizes the overall error. The gradients are calculated for each output neuron and for all training examples. The average of these gradients gives the desired weight changes for one training iteration.

3. **Weight Update**

The weights are updated using an optimization algorithm like gradient descent. The basic idea of gradient descent is to adjust the weights in the direction that minimizes the cost function fastest. The weight update is performed as follows: `weights = weights - gradient * learning_rate`. The learning rate is a hyperparameter that controls the size of the steps taken by the gradient descent algorithm. This process is iterative, meaning it repeats multiple times. With each iteration, the model's accuracy should improve as the weights are fine-tuned to better fit the data. Learning = finding weights and biases that minimize cost function

4. **Efficiency Considerations**

Calculating the gradient for every training example in every iteration can be computationally expensive. To address this, we often use mini-batch gradient descent, where the training data is divided into mini-batches and the gradient is computed for each mini-batch. This provides a good balance between computational efficiency and the accuracy of the gradient estimate. In stochastic gradient descent (SGD), the gradient is computed and the weights are updated for each training example (one example, not all). This can lead to faster convergence, but the path to the minimum can be more noisy compared to batch or mini-batch gradient descent.