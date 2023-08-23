import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(num_features + 1)  # initial random weights +1 for the bias term
        print(f'[INFO] \tRandom initial weights: {self.weights}')

    def predict(self, inputs): # activation function
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0] # activation potential: u
        return 1 if summation >= 0 else -1 # Use of the bipolar step function

    def train(self, training_data, labels):
        hasError = True
        for epoch in range(self.epochs):
            #print(f'[INFO] Epoch: {epoch}')
            hasError = False
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs) # return of activation function: y
                if prediction != label:
                    hasError = True
                    update = self.learning_rate * (label - prediction) # eta * (dk - y)
                    self.weights[1:] += update * inputs # update weights: w <- w + eta * (dk - y) * xk
                    self.weights[0] += update # update activation limiar: tetha <- tetha + eta * (dk - y)
                    #print(f'[INFO] Weights: {self.weights}')
            if hasError == False:
                print(f'[INFO] \tConverged after: {epoch + 1} epochs.')
                break
        print(f'[INFO] \tFinal weights: {self.weights}')
        print(f'[INFO] \tTotal of epochs: {epoch + 1}')

if __name__ == "__main__":
    print(f'\n[INFO] ###### Perceptron Implementation #######')
    print(f'\n[INFO] Loading training dataset and labels...')
    file = open('tab_treinamento1.dat', 'r')
    results = list()
    l = list()

    for line in file:
        columns = line.split()
        columns = np.array(columns, dtype=float)
        results.append(columns[:3])
        l.append(columns[-1:])
        
    training_data = np.array(results)
    labels = np.array(l)
    print(f'\t[INFO] OK!')

    #training_data = np.array([[0.6508, 0.1097, 4.0009], [-1.4492, 0.8896, 4.4005], [2.085, 0.6876, 1.2071], [0.2626, 1.1476, 7.7985], [0.6418, 1.0234, 7.0427], [0.2569, 0.673, 8.3265], [1.1155, 0.6043, 7.4446], [0.0914, 0.3399, 7.0677], [0.0121, 0.5256, 4.6316], [-0.0429, 0.466, 5.4323], [0.434, 0.687, 8.2287], [0.2735, 1.0287, 7.1934], [0.4839, 0.4851, 7.485], [0.4089, -0.1267, 5.5019], [1.4391, 0.1614, 8.5843], [-0.9115,  -0.1973, 2.1962], [0.3654, 1.0475, 7.4858], [0.2144, 0.7515, 7.1699], [0.2013, 1.0014, 6.5489], [0.6483, 0.2183, 5.8991], [-0.1147, 0.2242, 7.2435], [-0.797, 0.8795, 3.8762], [-1.0625, 0.6366, 2.4707], [0.5307, 0.1285, 5.6883], [-1.22, 0.7777, 1.7252], [0.3957, 0.1076, 5.6623], [-0.1013, 0.5989, 7.1812], [2.4482, 0.9455, 11.2095], [2.0149, 0.6192, 10.9263], [0.2012, 0.2611, 5.4631]])
    #labels = np.array([-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1])
    print(f'[INFO] Training dataset: \n{training_data}')
    print(f'[INFO] Labels of training dataset: \n{labels}')
    
    # Testing dataset
    # test_data = np.array([[0.6508, 0.1097, 4.0009], [-1.4492, 0.8896, 4.4005], [2.085, 0.6876, 1.2071], [0.2626, 1.1476, 7.7985], [0.6418, 1.0234, 7.0427], [0.2569, 0.673, 8.3265], [1.1155, 0.6043, 7.4446], [0.0914, 0.3399, 7.0677], [0.0121, 0.5256, 4.6316], [-0.0429, 0.466, 5.4323], [0.434, 0.687, 8.2287], [0.2735, 1.0287, 7.1934], [0.4839, 0.4851, 7.485], [0.4089, -0.1267, 5.5019], [1.4391, 0.1614, 8.5843], [-0.9115,  -0.1973, 2.1962], [0.3654, 1.0475, 7.4858], [0.2144, 0.7515, 7.1699], [0.2013, 1.0014, 6.5489], [0.6483, 0.2183, 5.8991], [-0.1147, 0.2242, 7.2435], [-0.797, 0.8795, 3.8762], [-1.0625, 0.6366, 2.4707], [0.5307, 0.1285, 5.6883], [-1.22, 0.7777, 1.7252], [0.3957, 0.1076, 5.6623], [-0.1013, 0.5989, 7.1812], [2.4482, 0.9455, 11.2095], [2.0149, 0.6192, 10.9263], [0.2012, 0.2611, 5.4631]])

    # Creating a Perceptron
    print(f'\n[INFO] Creating a Perceptron...')
    number_of_epochs = 10000
    perceptron = Perceptron(num_features=3, learning_rate=0.01, epochs=number_of_epochs)
    print(f'[INFO] \tOK!')

    # Training the perceptron
    print(f'\n[INFO] Getting information about training dataset...')
    print(f'[INFO] \tTraining dataset size = {training_data.shape[0]}')
    print(f'[INFO] \tLabels size = {labels.shape[0]}')
    print(f'[INFO] \tLimit of epochs: {number_of_epochs}')
    print(f'\n[INFO] Training the Perceptron...')
    perceptron.train(training_data, labels)
    print(f'\t[INFO] OK!')

    # Testing the perceptron
    print(f'\n[INFO] Loading testing dataset...')
    file = open('tab_teste1.dat', 'r')
    results = list()

    for line in file:
        columns = line.split()
        columns = np.array(columns, dtype=float)
        results.append(columns[:])
        
    testing_data = np.array(results)
    print(f'\t[INFO] OK!')

    print(f'\n[INFO] Getting information about testing dataset...')    
    # print(f'[INFO] \tTesting dataset size = {training_data.shape[0]}')
    print(f'[INFO] \tTesting dataset size = {testing_data.shape[0]}')
    print(f'\n[INFO] Running testing data...')
    # for inputs in training_data:
    for inputs in testing_data:
        result = perceptron.predict(inputs)
        print(f"[INFO] \tInput: {inputs} -> Output: {result}")
