# training via backpropgation 

def _rand_init_weights(self, size_in, size_out):
    return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]
        self.theta1 = self._rand_init_weights(400, num_hidden_nodes)
        self.theta2 = _rand_init_weights(num_hidden_nodes, 10)
        self.input_layer_bias = _rand_init_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_init_weights(1, 10)