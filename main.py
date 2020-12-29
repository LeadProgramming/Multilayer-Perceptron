import math
from random import random, randint, uniform, sample

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


def data_minmax(data):
    stats = [[min(col), max(col)] for col in zip(*data)]
    return stats

def normalize_data(data,minmax):
    for row in data:
        for i in range(len(row)):
            if(minmax[i][1] - minmax[i][0] != 0):
                row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])

def sigmoid(z):
    return 1.0/(1.0+math.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

class ANN(object):

    def __init__(self,layers):
        self.n_input = layers[0]
        self.n_hidden = layers[1]
        self.n_output = layers[2]
        #used to adjust hyperparemeters
        # self.weights = [list([[ uniform(-0.1,0.1) for i in range(self.n_input)] for i in range(self.n_hidden)]),
        #                 list([[ uniform(-0.1,0.1) for i in range(self.n_hidden)] for i in range(self.n_output)])]
        self.weights = [list([[0.05,-0.03,0.02,-0.01,0.05,0.01,0.02,0.08,-0.01,0.03],
                              [0.01,-0.02,-0.02,0.09,-0.03,0.02,0.06,0.05,0.06,0.04],
                              [-0.02,-0.08,-0.02,0.09,-0.03,0.06,0.06,0.03,-0.04,0.08],
                              [0.03,-0.07,-0.02,0.09,-0.02,0.08,0.06,0.02,0.02,0.07],
                              [-0.04,-0.06,-0.02,0.09,-0.09,0.09,0.06,0.01,-0.03,0.03],
                              [0.05,-0.02,-0.02,0.09,-0.06,-0.05,0.06,0.08,0.02,0.02],
                              [-0.08,-0.05,-0.03,0.04,-0.02,0.03,0.06,0.07,-0.05,0.01],
                              [0.02,-0.04,-0.07,0.09,-0.05,-0.05,0.06,0.05,0.02,0.09],
                              [0.05,-0.05,-0.04,0.03,-0.08,0.03,0.06,0.03,0.08,0.02],
                              [0.08,-0.06,-0.06,0.04,-0.09,0.04,0.02,0.01,-0.07,0.08]]),
                        list([[-0.04,-0.06,-0.02,0.09,-0.09,0.09,0.06,0.01,-0.03,0.03],
                              [0.03,-0.07, -0.01,0.04,-0.03,0.08,0.06,0.02,0.02,0.07],
                              [0.01,-0.02,-0.08,0.01,-0.05,0.02,0.01,0.08,0.06,0.04],
                              [0.09,-0.09,-0.09,0.02,-0.07,0.09,0.02,0.04,0.03,0.04],
                              [0.05,-0.04,-0.03,0.09,-0.03,0.07,0.03,0.09,0.06,0.04],
                              [0.07,-0.05,-0.06,0.03,-0.08,0.05,0.05,0.05,0.01,0.02],
                              [0.02,-0.08,-0.06,0.05,-0.05,0.03,0.03,0.09,0.02,0.02],
                              [0.01,-0.03,0.02,-0.01,0.05,0.01,0.02,0.08,-0.01,0.03]])]
     
    def softmax(self,wgts,hdn,total):
        return math.exp(dot(wgts,hdn))/total
        
    def feed_forward(self, a):
        """ Forward Propagation  """
        hidden_neurons = []
        for w in self.weights[0]:
            # transfer function -> activation function -> sigmoid neuron -> layer of neurons
            hidden_neurons.append(sigmoid(dot(w, a)))
        # needed for calculating the hidden deltas
        output_neurons = []
        
        total = sum([math.exp(dot(w,hidden_neurons)) for w in self.weights[1]])
        for w in self.weights[1]:
            # output_neurons.append(sigmoid(dot(w, hidden_neurons)))
            output_neurons.append(self.softmax(w, hidden_neurons,total))
        return hidden_neurons, output_neurons

    def backward_propagate(self, hidden, output, target):
        # output layer delta
        output_delta = []
        hidden_delta = []
        output_delta = [i*(1.0-i)*(j-i) for i, j in zip(output, target)]
        # transpose the weight vectors
        t_weights = map(list, zip(*self.weights[1]))
        # iterate the rows of transposed_weights and use the dot.
        swjidi = [dot(i, output_delta) for i in t_weights]
        # hidden layer delta
        hidden_delta = [i*(1.0-i)* j for i,j in zip(hidden,swjidi)]
        # [hidden,output]
        return hidden_delta, output_delta

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
        tp_weights = []
        for weight_list, delta in zip(self.weights[0], deltas[0]):
            tp_weight_list = []
            for weight, input_elem in zip(weight_list, input):
                tp_weight_list.append(weight+learning_rate*delta*input_elem)
            tp_weights.append(tp_weight_list)
        self.weights[0] = tp_weights

    def train_network(self, training_data, target_data, num_epoch, learning_rate=0.1):
        for epoch in range(num_epoch+1):
            annealing = learning_rate*math.exp(-0.0003*epoch)
            for training, target in zip(training_data, target_data):
                hidden, output = self.feed_forward(training)
                error = self.cost_func(output, target)
                deltas = self.backward_propagate(hidden, output, target)
                self.update_weights(hidden, deltas, training, annealing)
            print("epoch={0}, learning_rate={1}, error={2}".format(
                epoch, annealing, error))
        print("-----------NEW WEIGHTS------------")
        [print(j) for i in self.weights for j in i]

    def classify(self, row):
        output = self.feed_forward(row)
        return find_max(output[1])

    def calc_acc(self, X, target, pop_size):
        count = 0
        #uncomment to see the classifieds
        # print(X, target)
        for i, j in zip(X, target):
            if(i == j):
                count += 1
        return count/pop_size

    def cost_func(self, X, target):
        return sum([(y-x)**2.0 for x, y in zip(X, target)])/len(X)

def read_file(name):
    """
    reads the file's content line then splits the content which is delimited by commas.
    :param name: file's name
    :return:
    """
    data = []
    with open(name, 'r') as f:
        print("File: {0} is opened and being read.".format(name))
        for i in f:
            line = map(int, i.split(","))
            data.append(list(line))
        return data

if __name__ == "__main__":
    data = read_file("dataset.csv")
    train_set = data[:int(len(data)*0.8)]
    valid_set = data[int(len(data)*0.8):int(len(data)*0.9)]
    test_set = data[int(len(data)*0.9):]

    train_feats = [i[:-1] for i in train_set]
    train_labels = [i[-1] for i in train_set]

    valid_feats = [i[:-1] for i in valid_set]
    valid_labels = [i[-1] for i in valid_set]

    test_feats = [i[:-1] for i in test_set]
    test_labels = [i[-1] for i in test_set]
    
    # normalizing
    normalize_data(train_feats, data_minmax(train_feats))
    normalize_data(valid_feats, data_minmax(valid_feats))
    normalize_data(test_feats, data_minmax(test_feats))
    
    """ Hyperparameter
    Input Layer: 10 inputs
    Hidden Layer #1: 10 sigmoid neurons
    Output Layer: 8 output softmax neurons
    """
    net = ANN([10,10,8])

    #8 possible classes: 0 - 8
    vectorized_target = [vectorize(i, 8) for i in train_labels]
    net.train_network(train_feats, vectorized_target, 500, 0.5)
    train = [net.classify(i) for i in train_feats]
    valid = [net.classify(i) for i in valid_feats]
    test = [net.classify(i) for i in test_feats]
    print("TRAIN ACCURACY: ", net.calc_acc(
        train, train_labels, len(train_labels)))
    print("VALID ACCURACY: ", net.calc_acc(
        valid, valid_labels, len(valid_labels)))
    print("TEST ACCURACY: ", net.calc_acc(
        test, test_labels, len(test_labels)))