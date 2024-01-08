import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

training_outputs = np.array([[0],
                             [1],
                             [1],
                             [0]])

input_size = 2
hidden_size = 4
output_size = 1

hidden_weights = np.random.uniform(size=(input_size, hidden_size))
hidden_bias = np.random.uniform(size=(1, hidden_size))
output_weights = np.random.uniform(size=(hidden_size, output_size))
output_bias = np.random.uniform(size=(1, output_size))

learning_rate = 0.1
epochs = 10000

def mean_squared_error(predicted_output, target_output):
    return np.mean(np.square(predicted_output - target_output))

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_activation = sigmoid(np.dot(training_inputs, hidden_weights) + hidden_bias)
    output_layer_activation = sigmoid(np.dot(hidden_layer_activation, output_weights) + output_bias)

    # Calculate error using MSE
    error = mean_squared_error(output_layer_activation, training_outputs)
    
    # Backpropagation
    d_output = (output_layer_activation - training_outputs) * sigmoid_derivative(output_layer_activation)
    error_hidden_layer = d_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

    # Updating weights and biases
    output_weights -= hidden_layer_activation.T.dot(d_output) * learning_rate
    output_bias -= np.sum(d_output, axis=0, keepdims=True) * learning_rate
    hidden_weights -= training_inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias -= np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Testing the trained model
hidden_layer_activation = sigmoid(np.dot(training_inputs, hidden_weights) + hidden_bias)
output_layer_activation = sigmoid(np.dot(hidden_layer_activation, output_weights) + output_bias)

print("Output after training:")
print(output_layer_activation)
