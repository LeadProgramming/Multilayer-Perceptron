![mlp_anim.gif](https://github.com/LeadProgramming/Multilayer-Perceptron/blob/master/mlp_anim.gif?raw=true) 

# Multilayer Perceptron
Welcome to my Multilayer Perceptron back-propagation neural network. This is the last project assigned by my AI professor. This is my first time creating a neural network from scratch and it was a ton of fun.

### Current Accuracies

![rates.png](https://github.com/LeadProgramming/Multilayer-Perceptron/blob/master/rates.png?raw=true) 

### Hyperparameters
**Input layer**: 10 inputs

**Hidden Layer #1**: 10 sigmoid neurons

**Output layer**: 8 softmax neurons 

**Epoch**: 500

**Learning rate**: 0.5

**Adaptive learning rate**:   a = 0.0003

**Training Set**: 80%

**Validation Set**: 10%

**Test Set**: 10%

## Design

Weights are chosen from [0.1,-0.1].

The bias is trainable.

### 1. Preprocessing

Each column of the feature vector has a value ranging between [0,96].  The feature vector must be normalized between 0 and 1.

![img](https://www.statisticshowto.com/wp-content/uploads/2015/11/normalize-data.png)

The target vector has a value ranging between [0,7] which needs to be one-hot-encoded. Depending on whichever output neuron has the highest activation will have a value of one and the rest of the neuron will have a value of zero. 

Note: One-hot encoding the target vector is useful for computing the error in gradient descent. Without the one-hot encoding, the target vector is useful in computing the error rate.

e.g 

without one-hot encoding : [ 0.02, 0.03, 0.02, 0.10, 0.21, 0.2, 0.1, 0.9]

with one-hot encoding :       [ 0, 0, 0, 0, 0, 0, 0, 1] 

classified: 7

### 2. Feed-Forward

The hidden and output layer will have its own activation function and requires a transfer function. The transfer function will compute the weighted sums from the weights and input/activated values coming from the transmitting neurons.  Here is our transfer function.

![img](https://latex.codecogs.com/svg.latex?\sum_k%20w_{kj}x_k)

The **hidden layer** uses the **sigmoid function** for the **activation** function. The hidden layer is responsible for independently and linearly determining a class with the given inputs.  The transfer function is our x. 

Note: It would be useful if you store the activated values in somewhere in your code for the gradient descent algorithm later on. 

![S(t)= \frac {1}{1+e^{-t}}](https://www.gstatic.com/education/formulas2/-1/en/sigmoid_function.svg)

The classified set has eight possible classes which means that the random variable has a multinomial distribution. Thus, the activation function of the **output** layer uses a **SoftMax** regression.

![softmax.png](https://github.com/LeadProgramming/Multilayer-Perceptron/blob/master/softmax.svg?raw=true)

The result of the SoftMax function gives you a probability distribution for 8 possible classes. From there we select the class with the highest probability. 

Note: Getting the max value's index from the output layer is useful for obtaining the error rate. Not getting the max is useful for calculating the delta for every output neurons, because were leaving the value as is.

Here is the feed-forward formula which is used for classifying an example.

![img](https://latex.codecogs.com/svg.latex?y_i%20=%20%20f(\sum_j%20w_{ji}f(\sum_k%20w_{kj}x_k)))

### 3. Directional Error

> The neural network can only learn from its mistake. 

The **directional error** tells how bad the neural network is classifying an example. t is our target and y is what the model is classifying. This formula will be useful for gradient descent because it will adjust the weights depending on its performance.

![img](https://latex.codecogs.com/svg.latex?(t_i-y_i))

### 4. Gradient Descent

> Now we can penalize the mistake the neural net for making mistakes. 

Since we propagated from the input to the output layer, now we go backwards and calculate the errors (**delta**) for **each** neuron in **every** single layer. For example, if there are eight output neurons, then there will be eight deltas for the output layer. 

Note: Delta can also be negative which can decrease the weight change later on.

![img](https://latex.codecogs.com/svg.latex?\delta_i)

**Output-layer**

We take the non one-hot encoded result from the forward propagation and plug it into this formula.

Note: Since, the one-hot encoded target vector will have an output that has a value of one and the rest of the output neuron will have a value of zero, then there will be a delta that will have a positive value and the rest will have negative values.

![img](https://latex.codecogs.com/svg.latex?\delta_i%20=%20y_i(1-y_i)(t_i%20-%20y_i))

Note: the directional error informs us the performance of the classifier. 

![img](https://latex.codecogs.com/svg.latex?(t_i%20-%20y_i))

 The output gradient or the derivative of the SoftMax function will allow us to traverse the gradient and hopefully lands us on the global minima. 

![img](https://latex.codecogs.com/svg.latex?y_i(1-y_i))

Note: The global minima is when the error for each delta becomes so small there are barely any weight changes. Now, if you let neural network train a bit more the weight changes will increase thus overshooting the global minima and making more errors!

**Hidden-layer**

Now, we need to compute the deltas for the hidden neurons. 

![img](https://latex.codecogs.com/svg.latex?\delta_j%20=%20h_j(1-h_j)\sum_i%20\delta_i%20w_{ji})

The step is almost the same for the output neurons except we transpose the weight matrix for the output neurons. This step is very important because the dimensions for both the output delta vector and the weight matrix will not match! Python will still perform the dot product regardless of the mismatch in dimensions!

![img](https://latex.codecogs.com/svg.latex?(w_{ji})^T)

Now, the length of the row of deltas should match with the length of the row of the weights for a single hidden neuron. We can finally perform the dot product for every row in the output weight matrix. 

![img](https://latex.codecogs.com/svg.latex?\sum_i%20\delta_i%20w_{ji})

Interestingly, prior to transposing the weight matrix the length of a single row is actually the number of neurons in the hidden layer. This will determine the number of delta we will have for the hidden layer.

Let's not forget that we need to compute the hidden gradient.

![img](https://latex.codecogs.com/svg.latex?h_j(1-h_j))

Now, we needed the activated values from the hidden neurons which came from our feed-forward algorithm that you stored somewhere in your code.

We have calculated the deltas for the hidden layer and the output layer.

### 5. Updating the Weights

Time to change the weights! 

**Output-layer**

For each hidden neuron, every output delta, the corresponding weight, and the learning rate will change the output layer weights. Here is the formula to make more sense.

![img](https://latex.codecogs.com/svg.latex?w_{ji}%20=%20w_{ji}%20+%20\eta\delta_ih_j)

**Hidden-layer**

For each input, every hidden delta, the corresponding weight, and the learning rate will change the hidden layer weights. Here is the formula to make more sense.

![img](https://latex.codecogs.com/svg.latex?w_{kj}%20=%20w_{kj}%20+%20\eta\delta_jx_k)

### 6. Training the Network

Now, we want to repeat step 2 to 5 for all training examples which will train our data. After we train the rest of the training data we will have completed one **epoch**. Now we can choose the number of epochs to train our network!

**Annealing Learning Rate** - I found out this pretty neat trick to decrease the step size over a number of epochs. This trick is useful to hyper-jump to the global minimum and skip a bunch of local minimums. 

![img](https://latex.codecogs.com/svg.latex?\eta(t)%20=%20\eta(0)e^{-at})

### 7. Cost Function 

Note: At the end of each epoch we can take the mean of the square (directional) error (**MSE**). The MSE will tell us if we have reached the global minimum and we can tune the hyper-parameters to our liking. Also, the MSE tells us how well the classifier did against the target vector, hence if the classifier made a ton of mistakes it would be more "costly".

![img](https://latex.codecogs.com/svg.latex?MSE%20=%20\frac{1}{M}\sum_{i=1}^m(t_i%20-%20y_1)^2))

To compute the **error rate** on the training/validation/test set, we can just sum up the number of correct classification divided by the size of the dataset. 



## Remarks:

When encountering this project, I naively implemented sigmoid function on the output layer and learned a lot. SoftMax saved the day and gave me higher training accuracy. I was able to teach my neural network. The neural network did pretty average on its own. However, in the future I will need to implement a better validation technique like K-Folds or Leave-One-Out Cross Validation. 

After finishing the project, I also did research and realized MSE works pretty well for regression models. However, I would need to use cross-entropy to better penalize the mistakes my model made. This could potentially reduce the number of epochs needed to train the model and potentially have better accuracy. 

## Setup:

Prerequisites: Python, Excel, IDE (of your choice), PDF viewer

1. dataset.csv contains all of the data.
2. All of the pdf files contains more details about how the project should be orchestrated.
3. main.py contains all of the multilayer perception's source code. 

## Executing:
4. Type and enter the following code on a CLI:

   ```
   python main.py
   ```

   **or**

   Run it on an IDE of your choice.

   I used visual studio code.

## References:

Kubat, Miroslav. *Introduction To Machine Learning.* S.L., Springer International Pu, 2018.

“Unsupervised Feature Learning and Deep Learning Tutorial.” *Deeplearning.Stanford.Edu*, deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/.
