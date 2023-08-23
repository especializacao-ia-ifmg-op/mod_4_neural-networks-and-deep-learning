import numpy as np

class Adaline:
    def __init__(self, num_features, learning_rate=0.01, epochs=100, epsilon=1e-5):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.weights = np.random.rand(num_features + 1)  # initial random weights +1 for the bias term
        print(f'[INFO] \tRandom initial weights: {self.weights}')

    def predict(self, inputs): # activation function
        activation = np.dot(inputs, self.weights[1:]) + self.weights[0] # activation potential: u
        return 1 if activation >= 0 else -1  # Use of the bipolar step function

    def train(self, training_data, targets):
        mse = 0
        for epoch in range(self.epochs):
            total_error = 0
            #print(f'[INFO] Epoch: {epoch}')
            for inputs, target in zip(training_data, targets):
                activation = self.predict(inputs) # return of activation function: y
                error = target - activation
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
                total_error += error ** 2
            mse = total_error / len(targets)
            if mse < self.epsilon:
                print(f'[INFO] \tConverged after: {epoch + 1} epochs.')
                break
        print(f'[INFO] \tFinal weights: {self.weights}') 
        print(f'[INFO] \tTotal of epochs: {epoch + 1}')
        print(f'[INFO] \tMean square error: {mse}')


if __name__ == "__main__":
    print(f'\n[INFO] ###### Adaline Implementation #######')
    print(f'\n[INFO] Loading training dataset and targets...')
    file = open('tab_treinamento2.dat', 'r')
    results = list()
    t = list()

    for line in file:
        columns = line.split()
        columns = np.array(columns, dtype=float)
        results.append(columns[:4])
        t.append(columns[-1:])
        
    training_data = np.array(results)
    targets = np.array(t)
    print(f'[INFO] \tOK!')

    #training_data = np.array([[4.3290000e-01, -1.3719000e+00, 7.0220000e-01, -8.5350000e-01],[3.0240000e-01, 2.2860000e-01, 8.6300000e-01, 2.7909000e+00],[1.3490000e-01, -6.4450000e-01, 1.0530000e+00, 5.6870000e-01],[3.3740000e-01, -1.7163000e+00, 3.6700000e-01, -6.2830000e-01],[1.1434000e+00, -4.8500000e-02, 6.6370000e-01, 1.2606000e+00],[1.3749000e+00, -5.0710000e-01, 4.4640000e-01, 1.3009000e+00],[7.2210000e-01, -7.5870000e-01, 7.6810000e-01, -5.5920000e-01],[4.4030000e-01, -8.0720000e-01, 5.1540000e-01, -3.1290000e-01],[-5.2310000e-01, 3.5480000e-01, 2.5380000e-01, 1.5776000e+00],[3.2550000e-01, -2.0000000e+00, 7.1120000e-01  -1.1209000e+00],[5.8240000e-01, 1.3915000e+00, -2.2910000e-01, 4.1735000e+00],[1.3400000e-01, 6.0810000e-01, 4.4500000e-01, 3.2230000e+00],[1.4800000e-01, -2.9880000e-01, 4.7780000e-01, 8.6490000e-01],[7.3590000e-01, 1.8690000e-01, -8.7200000e-02, 2.3584000e+00],[7.1150000e-01, -1.1469000e+00, 3.3940000e-01, 9.5730000e-01],[8.2510000e-01, -1.2840000e+00, 8.4520000e-01, 1.2382000e+00],[1.5690000e-01, 3.7120000e-01, 8.8250000e-01, 1.7633000e+00],[3.3000000e-03, 6.8350000e-01, 5.3890000e-01, 2.8249000e+00],[4.2430000e-01, 8.3130000e-01, 2.6340000e-01, 3.5855000e+00],[1.0490000e+00, 1.3260000e-01, 9.1380000e-01, 1.9792000e+00],[1.4276000e+00, 5.3310000e-01, -1.4500000e-02, 3.7286000e+00],[5.9710000e-01, 1.4865000e+00, 2.9040000e-01, 4.6069000e+00],[8.4750000e-01, 2.1479000e+00, 3.1790000e-01, 5.8235000e+00],[1.3967000e+00, -4.1710000e-01, 6.4430000e-01, 1.3927000e+00],[4.4000000e-03, 1.5378000e+00, 6.0990000e-01, 4.7755000e+00],[2.2010000e-01, -5.6680000e-01, 5.1500000e-02, 7.8290000e-01],[6.3000000e-01, -1.2480000e+00, 8.5910000e-01, 8.0930000e-01],[-2.4790000e-01, 8.9600000e-01, 5.4700000e-02, 1.7381000e+00],[-3.0880000e-01, -9.2900000e-02, 8.6590000e-01, 1.5483000e+00],[-5.1800000e-01, 1.4974000e+00, 5.4530000e-01, 2.3993000e+00],[6.8330000e-01, 8.2660000e-01, 8.2900000e-02, 2.8864000e+00],[4.3530000e-01, -1.4066000e+00, 4.2070000e-01, -4.8790000e-01],[-1.0690000e-01, -3.2329000e+00, 1.8560000e-01, -2.4572000e+00],[4.6620000e-01, 6.2610000e-01, 7.3040000e-01, 3.4370000e+00], [8.2980000e-01, -1.4089000e+00, 3.1190000e-01, 1.3235000e+00]])
    # targets = np.array([1.0000000e+00, -1.0000000e+00, -1.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00, -1.0000000e+00, 1.0000000e+00, -1.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00, -1.0000000e+00, -1.0000000e+00, 1.0000000e+00, -1.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00, -1.0000000e+00, -1.0000000e+00, 1.0000000e+00, -1.0000000e+00, 1.0000000e+00, -1.0000000e+00, 1.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00, -1.0000000e+00, -1.0000000e+00, -1.0000000e+00])
    print(f'[INFO] \tTraining dataset: \n{training_data}')
    print(f'[INFO] \tTargets of training dataset: \n{targets}')
    
    # Testing dataset
    # test_data = np.array([[-0.3665, 0.062, 5.9891], [-0.7842, 1.1267, 5.5912], [0.3012, 0.5611, 5.8234], [0.7757, 1.0648, 8.0677], [0.157, 0.8028, 6.304], [-0.7014, 1.0316, 3.6005], [0.3748, 0.1536, 6.1537], [-0.692, 0.9404, 4.4058], [-1.397, 0.7141, 4.9263], [-1.8842, -0.2805, 1.2548]])

    # Creating an Adaline
    print(f'\n[INFO] Creating an Adaline...')
    number_of_epochs = 10000
    epsilon = 1e-5
    adaline = Adaline(num_features=4, learning_rate=0.01, epochs=number_of_epochs, epsilon=epsilon)
    print(f'[INFO] \tOK!')    

    # Training the Adaline
    print(f'\n[INFO] Getting information about training dataset...')
    print(f'[INFO] \tTraining dataset size = {training_data.shape[0]} {training_data.shape[1]}')
    print(f'[INFO] \tLabels size = {targets.shape[0]} {targets.shape[1]}')
    print(f'[INFO] \tLimit of epochs: {number_of_epochs}')
    print(f'[INFO] \tEpsilon: {epsilon}')
    print(f'\n[INFO] Training the Adaline...')
    adaline.train(training_data, targets)
    print(f'[INFO] \tOK!')

    # # Testing the adaline
    print(f'\n[INFO] Loading testing dataset...')
    file = open('tab_teste2.dat', 'r')
    results = list()

    for line in file:
        columns = line.split()
        columns = np.array(columns, dtype=float)
        results.append(columns[:])
        
    testing_data = np.array(results)
    print(f'\t[INFO] OK!')

    print(f'\n[INFO] Getting information about testing dataset...')
    # print(f'[INFO] Testing dataset size = {training_data.shape[0]}')
    print(f'[INFO] \tTesting dataset size = {testing_data.shape[0]}')
    print(f'\n[INFO] Running testing data...')
    # for inputs in training_data:
    for inputs in testing_data:
        result = adaline.predict(inputs)
        print(f"[INFO] \tInput: {inputs} -> Output: {result}")
