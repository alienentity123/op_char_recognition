import numpy as np

# Dummy data for demonstration (replace with real data loading)
data_matrix = np.random.randint(0, 2, (100, 400))  # 100 samples, 400 features each
data_labels = np.random.randint(0, 10, 100)        # 100 labels (digits 0-9)

# Split indices into train and test
indices = np.arange(len(data_matrix))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_indices = indices[:split]
test_indices = indices[split:]

# try for various number of hidden nodes and see what performs the best 
for i in range(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    perform = str(test(data_matrix, data_labels, nn))
    print ("{i} Hidden Nodes:{val}".format(i=i, val=perform))

def test(data_matrix, data_labels, nn):
    avg_sum = 0
    for j in range(100):
        correct_guess_count = 0
        for i in test_indices: 
            test = data_matrix[i]
            predict = nn.predict(test)
            if data_labels[i] == predict:
                correct_guess_count += 1
        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 100 


    
