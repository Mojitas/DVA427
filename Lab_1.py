from Mathias_test import *
#from Dan_test import *
# TODO: Det går att skriva saker som måste göras på detta sätt. Då dyker de up under TODO-fliken

class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        self.w1 = 2 * np.random.random((19, 9)) - 1  # hidden layer 1
        self.w2 = 2 * np.random.random((9, 9)) - 1  # hidden layer 2
        self.w3 = 2 * np.random.random((9, 1)) - 1  # output layer
        self.learning_rate = 0.1
        self.l1 = np.zeros((1, 9))
        self.l2 = np.zeros((1, 9))

        self.bias1 = 2 * np.random.random((9, 1)) - 1  # Bias for hidden layer 1
        self.bias2 = 2 * np.random.random((9, 1)) - 1  # Bias for hidden layer 2
        self.bias3 = 2 * np.random.random((1, 1)) - 1  # Bias for output layer

    # commit lots of math
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  #

    def forward(self, input_layer):  # functions that uses more layers
        input_layer = input_layer.astype(float)
        self.l1 = self.sigmoid(np.dot(input_layer, self.w1))
        self.l2 = self.sigmoid(np.dot(self.l1, self.w2))
        output = np.dot(self.l2, self.w3)
        print(self.l1.shape)
        self.l2 = self.sigmoid(np.dot(self.l1, self.w2))
        output = np.dot(self.l2, self.w3)
        return self.sigmoid(output)

    def backwards(self, input_layer, output_layer, training_iterations):

        output = self.forward(input_layer)
        out_error = output_layer - output
        delta_1 = out_error * self.sigmoid_derivative(output)
        print("delta 1: {}\noutput: {}\nw3: {}".format(delta_1.shape, output.shape, self.w3.shape))
        self.w3 += self.learning_rate * output * delta_1
        delta_1 = out_error*self.sigmoid_derivative(output)
        self.w3 += self.learning_rate*self.l2*delta_1
        self.bias3 = self.learning_rate * delta_1

        # Allt är rätt hit tror jag
        delta_2 = np.multiply(self.l2,self.w3.T)*delta_1   #Vi tar ju redan sigmoid när vi sätter l2 i forward
        self.w2 += self.learning_rate*np.dot(self.l1.T,delta_2) #Kanske inte klar
        print(np.dot(self.l1.T,delta_2))
        self.bias2 = self.learning_rate * delta_2

        delta_3 = np.matmul(self.sigmoid_derivative(self.l1),self.w1) * delta_2    #TODO fixa den här
        self.w1 += self.learning_rate*np.dot(self.l1.T,delta_3)     #Ska använda inputs
        self.bias3 = self.learning_rate * delta_3

    def compare(self, inputs, output):  # func for comparing when training has been done
        inputs = inputs.astype(float)
        outputs = self.think(inputs)
        accuracy = 0
        size = inputs.shape[0]
        for i in range(size):
            if abs(output[i] - outputs[i]) < 0.1:
                accuracy += 1

        return accuracy / size


if __name__ == '__main__':
    training_iterations = 100
    DM.segmentation()  # Imports and sorts data
    NN = NeuralNetwork()

training_sessions = 0
NN.forward(DM.training_inputs[0, :])


while 0:
    i=10

    NN.train(DM.training_inputs, DM.training_outputs, i)
    accuracy = NN.compare(DM.validation_data, DM.validation_result)
    print("Overall accuracy is: {}\n training: {}".format(accuracy, training_sessions))
