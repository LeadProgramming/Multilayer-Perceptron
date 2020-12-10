import math
import random


def dot(x, y):
    if type(x) is not list:
        x = [x]
    if type(y) is not list:
        y = [y]
    return sum([i*j for i in x for j in y], 0)

def classify(x):
    return x.index(max(x))

def vectorize(x,size):
    tmp = [0] * size
    tmp[x] = 1
    return tmp


def scale(x, factor):
    return [i/factor for i in x]


def sigmoid(z):
    return 1.0/(1.0+math.exp(-z))


def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))


class ANN(object):

    def __init__(self, layers):
        self.layers_num = len(layers)
        self.neurons_num = layers
        self.bias = [list([.50, .65, .25,-.955]),
                     list([-.923, .357, -.663,.256,.378,.585,.152,.108])]
        self.weights = [list([[0.14, -0.45, -0.12,-0.12,0.89,-0.45,-0.16,0.56,-0.45,0.87],
                              [0.12, 0.23, -0.15,0.62,-0.821,0.512,-0.123,0.321,-0.754,0.831],
                              [0.64, -0.52, 0.24,0.93,-0.84,-0.64,0.52,-0.34,0.52,-0.89],
                              [-0.52, -0.12, -0.84,0.51,0.321,-0.123,0.721,0.632,-0.912,0.999]]),
                        list([[-0.123, -0.423, -0.321,-0.523],
                              [-0.412, -0.921, -0.731,0.213],
                              [-0.231, -0.421, 0.641,-0.592],
                              [0.521, 0.421, -0.781,-0.012],
                              [0.412,-0.891,0.512,-0.321],
                              [-0.721,0.951,0.213,-0.555],
                              [-0.812,0.231,-0.567,0.888],
                              [0.512,0.231,-0.751,-0.421]])]

    def feed_forward(self, a):
        """ Forward Propagation 
            Activation function """
        for i in range(self.layers_num-1):
            tmp = []
            for b, w in zip(self.bias[i], self.weights[i]):
                tmp.append(sigmoid(dot(w, a)+b))
            a = tmp
        return a
    
    def calc_error(self, X, target,pop_size):
        count = 0
        for i,j in zip(X,target):
            if(i == j ):
                count += 1
                print(i, j, count)
        return 1-(count/pop_size)
    
    def cost_func(self, X, target):
        #vectorized target values
        tmp = [0] * len(X)
        tmp[target] = 1
        return (sum([(y-x)**2 for x,y in zip(X,tmp)])/len(X))
        #return len(X)*(sum([(y-x)**2 for x,y in zip(X,tmp)]))
    # def cost_func(self,X,target):
    #     """ MSE """
    #     y = self.feed_forward(X)
    #     delta_1 = y(1-y)(target-y)
    #     #delta_2 = 
    #     return delta_1#, delta_2

  # def gradient_desc(self, training_data, eta = 0.01, epochs):


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
    # 80 - 20 split
    training_features = feature_vectors[:int(len(feature_vectors)*0.8)]
    training_labels = labels[:int(len(labels)*0.8)]

    test_features = feature_vectors[int(len(feature_vectors)*0.8):]
    test_labels = labels[int(len(labels)*0.8):]

    training_features = [scale(i, 100) for i in training_features]
    test_features = [scale(i, 100) for i in test_features]

    """ Hyperparameter
    Input Layer: 10 neurons
    Hidden Layer #1: 4 sigmoid neurons
    Output Layer: 4 output sigmoid neurons
    """

    net = ANN([10, 4, 4])
    
    output_training = [net.feed_forward(i) for i in training_features]
    classified_training = [classify(i) for i in output_training]
    vectorized_training = [vectorize(i, len(output_training[0])) for i in classified_training]
    vectorized_target = [vectorize(i, len(output_training[0])) for i in training_labels]
    error_rate = net.calc_error(vectorized_training, vectorized_target,len(data))
    print(error_rate)


    #prints out cost function for every classified training examples.
    #print([(net.cost_func(i,j)) for i, j in zip(classified_training, training_labels)])