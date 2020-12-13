import math
import random


def dot(x, y):
    if type(x) is not list:
        x = [x]
    if type(y) is not list:
        y = [y]
    return sum([i*j for i, j in zip(x, y)], 0)


def find_max(x):
    return x.index(max(x))

# one hot encode our data


def vectorize(x, size):
    tmp = [0] * size
    tmp[x] = 1
    return tmp


def scale(x, factor):
    return [i/factor for i in x]


def sigmoid(z):
    return 1.0/(1.0+math.exp(-z))


def sigmoidPrime(z):
    return sigmoid(z)*(1.0-sigmoid(z))


class ANN(object):

    def __init__(self, layers):
        self.layers_num = len(layers)
        self.neurons_num = layers
        self.bias = [list([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1]),
                     list([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])]
        self.weights = [list([[.3, .1, 0.12, 0.12, 0.1, 0.15, 0.16, 0.16, 0.15, 0.13],
                              [0.12, 0.23, 0.15, 0.12, 0.121,
                                  0.112, 0.123, 0.121, 0.12, .1],
                              [0.14, 0.12, 0.14, 0.13, 0.14,
                                  0.14, 0.12, 0.14, 0.2, 0.3],
                              [0.12, 0.12, 0.14, 0.31, 0.21,
                                  0.123, 0.121, 0.132, 0.12, 0.1],
                              [0.11, 0.14, 0.14, 0.21, 0.1, 0.123,
                                  0.121, 0.132, 0.15, 0.233],
                              [0.12, 0.13, 0.14, 0.11, 0.1, 0.123,
                                  0.221, 0.132, 0.15, 0.13],
                              [0.11, 0.13, 0.14, 0.21, 0.1, 0.123,
                                  0.121, 0.132, 0.15, 0.233],
                              [0.12, 0.11, 0.24, 0.31, 0.2, 0.123,
                                  0.321, 0.132, 0.15, 0.13],
                              [.2, .15, 0.12, 0.12, 0.19, 0.25,
                                  0.16, 0.16, 0.112, 0.13],
                              [0.12, 0.23, 0.15, 0.22, 0.221,
                                  0.112, 0.123, 0.221, 0.1, .2],
                              [0.14, 0.12, 0.24, 0.13, 0.24,
                                  0.24, 0.12, 0.14, 0.1, 0.1],
                              [0.12, 0.12, 0.24, 0.255, 0.121, 0.123, 0.121, 0.132, 0.5, 0.125]]),
                        list([[0.123, 0.21, 0.21, 0.123, 0.112, 0.251, 0.231, 0.265, 0.123, 0.21, 0.21, 0.123],
                              [0.012, 0.31, 0.131, 0.265, 0.123, 0.21,
                                  0.21, 0.123, 0.112, 0.288, 0.131, 0.165],
                              [0.231, 0.21, 0.141, 0.0852, 0.112, 0.01,
                                  0.231, 0.265, 0.231, 0.221, 0.141, 0.1852],
                              [0.121, 0.21, 0.281, 0.112, 0.221, 0.251,
                                  0.213, 0.155, 0.212, 0.191, 0.112, 0.221],
                              [0.212, 0.191, 0.112, 0.221, 0.231, 0.221,
                                  0.241, 0.0852, 0.021, 0.21, 0.281, 0.112],
                              [0.221, 0.051, 0.013, 0.055, 0.231, 0.221,
                                  0.241, 0.0852, 0.221, 0.051, 0.213, 0.255],
                              [0.112, 0.131, 0.167, 0.218, 0.212, 0.241,
                                  0.112, 0.121, 0.112, 0.231, 0.167, 0.188],
                              [0.112, 0.231, 0.051, 0.121, 0.112, 0.191, 0.012, 0.121, 0.212, 0.231, 0.251, 0.121]])]

    def feed_forward(self, a):
        """ Forward Propagation  """
        hidden_neurons = []
        for b, w in zip(self.bias[0], self.weights[0]):
            # transfer function -> activation function -> sigmoid neuron -> layer of neurons
            hidden_neurons.append(sigmoid(dot(w, a)))
        # needed for calculating the hidden deltas
        output_neurons = []
        for b, w in zip(self.bias[1], self.weights[1]):
            output_neurons.append(sigmoid(dot(w, hidden_neurons)))
        # print(hidden_neurons,output_neurons)
        return hidden_neurons, output_neurons

    def backward_propagate(self, hidden, output, target):
        deltas = []
        # output layer delta
        output_delta = [i*(1.0-i)*(j-i) for i, j in zip(output, target)]
        # insert the delta in front to index backwards
        deltas.insert(0, output_delta)

        # transpose the weight vectors
        t_weights = list(zip(*self.weights[1]))
        # named after the greatest wizard. :D
        swjidi = [dot(j, k)
                  # iterate the rows, iterate through the elments
                  for i in t_weights for j, k in zip(output_delta, i)]
        # hidden layer delta
        deltas.insert(0, [i*(1-i)*j for i, j in zip(hidden, swjidi)])
        # [ print(deltas) for i in deltas]
        # [hidden,output]
        return deltas

    def update_weights(self, hidden, deltas, input, learning_rate):
        # Change the weights of the output layer.
        tmp_weights = []
        # loop through a list of weight list and delta values.
        for weight_list, delta in zip(self.weights[1], deltas[1]):
            tmp_weight_list = []
            # loop through a list of weights and hidden layer neurons.
            for weight, layer_elem in zip(weight_list, hidden):
                tmp_weight_list.append(weight+learning_rate*delta*layer_elem)
            tmp_weights.append(tmp_weight_list)
        # we get a brand new weight changes for the hidden layer.
        self.weights[1] = tmp_weights

        # Change the weights of the hidden layer.
        # almost same implementation as above.
        tmp_weights = []
        for weight_list, delta in zip(self.weights[0], deltas[0]):
            tmp_weight_list = []
            for weight, input_elem in zip(weight_list, input):
                tmp_weight_list.append(weight+learning_rate*delta*input_elem)
            tmp_weights.append(tmp_weight_list)
        self.weights[0] = tmp_weights

    def train_network(self, training_data, target_data, num_epoch, learning_rate=0.1):
        for epoch in range(num_epoch):
            annealing = learning_rate*math.exp(-0.005*epoch)
            for training, target in zip(training_data, target_data):
                hidden, output = self.feed_forward(training)
                error = self.cost_func(output, target)
                deltas = self.backward_propagate(hidden, output, target)
                self.update_weights(hidden, deltas, training, annealing)
            print("epoch={0}, learning_rate={1}, error={2}".format(
                epoch, annealing, error))
        [print(j) for i in self.weights for j in i]

    def classify(self, row):
        output = self.feed_forward(row)
        return find_max(output[1])

    def calc_error(self, X, target, pop_size):
        count = 0
        print(X, target)
        for i, j in zip(X, target):
            if(i == j):
                print(i)
                count += 1
        return 1.0-(count/pop_size)

    def cost_func(self, X, target):
        return sum([(y-x)**2.0 for x, y in zip(X, target)])/len(X)

# lets find out how imbalanced our classes are!


def class_distribution(classes):
    count = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
    for i in classes:
        if(i == 0):
            count['0'] += 1
        elif(i == 1):
            count['1'] += 1
        elif(i == 2):
            count['2'] += 1
        elif(i == 3):
            count['3'] += 1
        elif(i == 4):
            count['4'] += 1
        elif(i == 5):
            count['5'] += 1
        elif(i == 6):
            count['6'] += 1
        elif(i == 7):
            count['7'] += 1
    print("CLASS DISTRIBUTION:", count)


def read_file(name):
    """
    reads the file's content line then splits the content which is delimited by commas.
    :param name: file's name
    :return:
    """
    data = []
    with open(name, 'r') as f:
        print("File: {0} is opened and being read.".format(name))
        # first row is the category name (ignored).
        f.readline().rstrip("\n").split(",")
        # second row and beyond are examples.
        for i in f:
            line = map(int, i.split(","))
            data.append(list(line))
        return data


""" def write_file(name):
"""     """
    :param name: name of a result's file
    :return:
    """ """
    with open(name, 'w') as f:
        print("File is written to {0}".format(name))
        f.writelines("w0: {0} \nw1: {1}".format(self.w0, self.w1)) """


if __name__ == "__main__":
    data = read_file("dataset.csv")
    feature_vectors = [i[1:len(data[0])-1] for i in data]
    labels = [i[len(data[0])-1] for i in data]
    class_distribution(labels)
    # normalizing
    feature_vectors = [scale(i, 96) for i in feature_vectors]
    # 80 - 20 split
    training_features = feature_vectors[:int(len(feature_vectors)*0.8)]
    training_labels = labels[:int(len(labels)*0.8)]

    # 10 - 10 split validation and test
    test_features = feature_vectors[int(len(feature_vectors)*0.8):int(len(feature_vectors)*0.9)]
    test_labels = labels[int(len(labels)*0.8):int(len(labels)*0.9)]

    validation_features = feature_vectors[int(len(feature_vectors)*0.9):]
    validation_labels = labels[int(len(labels)*0.9):]

    """ Hyperparameter
    Input Layer: 10 neurons
    Hidden Layer #1: 12 sigmoid neurons
    Output Layer: 8 output sigmoid neurons
    """
    net = ANN([10, 12, 8])

    # 8 possible classes: 0 - 7
    vectorized_target = [vectorize(i, 8) for i in training_labels]
    net.train_network(training_features, vectorized_target, 100, 2)

    validation = [net.classify(i) for i in validation_features]
    print("VALIDATION ERROR RATE: ", net.calc_error(validation, validation_labels, len(validation_labels)))
    #vectorized_test_labels = [vectorize(i, 8) for i in test_labels]
