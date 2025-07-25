import numpy as np
import math
import json

class OCRNeuralNetwork:
    NN_FILE_PATH = 'ocr_nn.json'
    LEARNING_RATE = 0.1

    def __init__(self, num_hidden_nodes, data_matrix=None, data_labels=None, train_indices=None, use_file=False):
        self.num_hidden_nodes = num_hidden_nodes
        self.input_size = 400  # assuming 20x20 input images
        self.output_size = 10  # digits 0-9
        self._use_file = use_file
        self.theta1 = self._rand_init_weights(self.input_size, num_hidden_nodes)
        self.theta2 = self._rand_init_weights(num_hidden_nodes, self.output_size)
        self.input_layer_bias = self._rand_init_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_init_weights(1, self.output_size)
        if use_file:
            self._load()
        if data_matrix is not None and data_labels is not None and train_indices is not None:
            self.train(data_matrix, data_labels, train_indices)

    def _rand_init_weights(self, size_in, size_out):
        return np.random.rand(size_out, size_in) * 0.12 - 0.06

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_prime(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def train(self, data_matrix, data_labels, train_indices):
        for idx in train_indices:
            data = {'y0': data_matrix[idx], 'label': data_labels[idx]}
            # Forward pass
            y0 = np.array(data['y0']).reshape(1, -1)  # shape (1, 400)
            y1 = np.dot(y0, self.theta1.T) + self.input_layer_bias  # (1, hidden)
            sum1 = y1
            y1 = self._sigmoid(y1)
            y2 = np.dot(y1, self.theta2.T) + self.hidden_layer_bias  # (1, 10)
            y2 = self._sigmoid(y2)
            # Backpropagation
            actual_vals = np.zeros((1, 10))
            actual_vals[0, data['label']] = 1
            output_errors = actual_vals - y2  # (1, 10)
            hidden_errors = np.dot(output_errors, self.theta2) * self._sigmoid_prime(sum1)  # (1, hidden)
            # Update weights and biases
            self.theta2 += self.LEARNING_RATE * np.dot(output_errors.T, y1)
            self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors.T, y0)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y0 = np.array(test).reshape(1, -1)
        y1 = np.dot(y0, self.theta1.T) + self.input_layer_bias
        y1 = self._sigmoid(y1)
        y2 = np.dot(y1, self.theta2.T) + self.hidden_layer_bias
        y2 = self._sigmoid(y2)
        results = y2.flatten().tolist()
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return
        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return
        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = np.array(nn['theta1'])
        self.theta2 = np.array(nn['theta2'])
        self.input_layer_bias = np.array(nn['b1'])
        self.hidden_layer_bias = np.array(nn['b2'])
            