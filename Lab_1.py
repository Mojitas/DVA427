from Mathias_test import *
#from Dan_test import *

class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        self.w1 = 2 * np.random.random((19, 9)) - 1   # weight matrix 1
        self.w2 = 2 * np.random.random((9, 9)) - 1  # weight matrix 2
        self.w3 = 2 * np.random.random((9, 1)) - 1  # weight matrix 3
        self.learning_rate = 0.1
        self.l1 = np.zeros((0, 0))  #hidden layer 1
        self.l2 = np.zeros((0, 0))  # hidden layer 2

        self.bias1 = 2 * np.random.random((1, 9)) - 1  # Bias for hidden layer 1
        self.bias2 = 2 * np.random.random((1, 9)) - 1  # Bias for hidden layer 2
        self.bias3 = 2 * np.random.random((1, 1)) - 1  # Bias for output layer

    # commit lots of math
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  #

    def forward(self, input_layer):  # functions that uses more layers

        input_layer = input_layer.astype(float)
        self.l1 = self.sigmoid(np.dot(input_layer, self.w1)) #+ self.bias1
        self.l2 = self.sigmoid(np.dot(self.l1, self.w2)) #+ self.bias2
        #print(self.sigmoid(np.dot(self.l2, self.w3) + self.bias3).shape)
        return self.sigmoid(np.dot(self.l2, self.w3))#+ self.bias3

    def backwards(self, input_layer, output_layer, training_iterations):

        for i in range(training_iterations):
            output = self.forward(input_layer)
            out_error = (output_layer - output)

            #print("Output is: ",output)
            delta_1 = out_error*self.sigmoid_derivative(output)
            #print("shape of l2: {}\nshape of delta1: {}\nshape of w3: {}".format(self.l2.shape, delta_1.shape, self.w3.shape))
            self.bias3 = self.learning_rate * delta_1
            #print("Shape of bias3: {}\nShape of w3: {}".format(self.bias3.shape,self.w3.shape))
            self.w3 += self.learning_rate*np.dot(self.l2.T,delta_1)

            delta_2 = np.multiply(self.sigmoid_derivative(self.l2),self.w3.T)*delta_1
            #print("shape of l2: {}\nshape of delta2: {}\nshape of w2: {}".format(self.l2.shape, delta_2.shape, self.w2.shape))
            self.bias2 = self.learning_rate * delta_2
            self.w2 += self.learning_rate*np.dot(self.l1.T,delta_2) #Kanske inte klar

            #delta_3 = np.dot(self.sigmoid_derivative(self.l1),self.w2.T) * delta_2
            #TODO Fixa denna for i: dot(sef.w2[i,:], delta2) sen multiply(sigmoid der l1, det)
            #print("\nshape of input: {}\nshape of delta3: {}\nshape of w1: {}".format(input_layer.shape, delta_3.shape,self.w1.shape))
            downstream = np.dot(self.w2, delta_2.T)

            delta_3 = np.multiply(NN.sigmoid_derivative(self.l1).T, downstream)



            self.bias3 = self.learning_rate * delta_3
            #print("shape of input layer: ",input_layer.shape)
            self.w1 += self.learning_rate*np.dot(input_layer.T,delta_3.T)     #Ska använda inputs


    def compare(self, inputs, output):  # func for comparing when training has been done
        inputs = inputs.astype(float)
        outputs = self.forward(inputs)
        accuracy = 0
        size = inputs.shape[0]
        for i in range(size):
            #print("shape of output: {}\nshape of outputs: {}".format(output[i,:].shape,outputs[i,:].shape))
            # TODO fixa rätt format
            if abs(output[i] - outputs[i]) < 0.5:
                accuracy += 1

        return accuracy / size


if __name__ == '__main__':
    DM.segmentation()  # Imports and sorts data
    NN = NeuralNetwork()

    training_sessions = 0
    iterations = 10000
    #group
    for i in range(iterations):
        NN.backwards(DM.training_inputs[i%864:(i+1)%864], DM.training_outputs[i%864:(i+1)%864], 10) # TODO fixa så det går att skicka in olika storlekar, just nu blir formatet konstigt
        #NN.backwards(DM.training_inputs[i,:], DM.training_outputs[i,:], 1)

        training_accuracy = NN.compare(DM.training_inputs,DM.training_outputs)
        validation_accuracy = NN.compare(DM.validation_data, DM.validation_result)
        test_accuracy = NN.compare(DM.test_data,DM.test_result)
        training_sessions += 1
        if i % 100 == 0:
            print("Training accuracy is: {}\nValidation accuracy is: {}\nTest accuracy is: {}\n training: {}".format(training_accuracy,validation_accuracy,test_accuracy, training_sessions))
