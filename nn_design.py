# try for various number of hidden nodes and see what performs the best 
for i in range(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    perform = str(test(data_matrix, data_labels, nn))
    print ("{i} Hidden Nodes:{val}".format(i=i, val=perform))

def test(data_matrix, data_labels,nn):
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


    
