# training via backpropgation 

def _rand_init_weights(self, size_in, size_out):
    return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]
    self.theta1 = self._rand_init_weights(400, num_hidden_nodes)
    self.theta2 = _rand_init_weights(num_hidden_nodes, 10)
    self.input_layer_bias = _rand_init_weights(1, num_hidden_nodes)
    self.hidden_layer_bias = self._rand_init_weights(1, 10)
    
# the sigmoid activation function. operates on scalars 
def _sigmoid_scalar(self, z):
    return 1 / (1 + mat.e ** -z)
    y1 = np.dot(np.mat(self.thetal), np.mat(data['y0']).T)
    sum1 = y1 + np.mat(self.input_layer_bias) #add the bias 
    y1 = self.sigmoid(sum1)

    y2 = np.dot(np.array(self.theta2), y1)
    y2 = np.add(y2, self.hidden_layer_bias) # add the bias 
    y2 = self.sigmoid(y2)

    #back propagation 
    actual_vals = [0] * 10
    actual_vals[data['label']] = 1
    output_errors = np.mat(actual_vals).T - np.mat(y2)
    hidden_errors = np.multiply(np.dot(np.mat(self.theta2).T, output_errors),
                                self_sigmoid_prime(sum1))
    
    self.theta1 += self.LEARNING_RATE * np.dot(np.mat(hidden_errors),np.mat(data['y0']))
    self.thta2 += self.LEARNING_RATE * np.dot(np.mat(output_errors),np.mat(y1).T)

    self.hidden_layer_bias += self.LEARNING_RATE * output_errors
    self. input_layer_bias += self.LEARNING_RATE * hidden_errors

    #testing a trained network 
    def predict(self, test):
        y1 = np.dot(np.mat(self.theta1), np.mat(test).T)
        y1 = y1 + np.mat(self.input_layer_bias) # adding the bias 
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.arrat(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias) # adding bias 
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))
    
    def save(self):
        if not self._use_file:
            return
        
        json_neural_network = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0],
            "b2": self.hidden_layer_bias[0].tolist()[0]
        };
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile: 
            json.dump(json_neural_network_nnFile)
    
    def _load(self):
        if not self._use_file:
            return
        
        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)

        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]
            