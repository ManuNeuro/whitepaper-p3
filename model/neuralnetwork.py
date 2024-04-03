import numpy as np

class NeuralNetwork:
    def __init__(self, inputSize, hiddenSize, outputSize, nbHiddenLayer):
        
        # Parameter of the network
        self.nbHiddenLayer = nbHiddenLayer
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.__version__ = 'NeuralNetwork-v0.1'
        
        # Initializing layers        
        self.inputs = np.zeros(inputSize)
        self.outputs = np.zeros(outputSize)
        self.hiddenLayer = [];
        for l in range(self.nbHiddenLayer):
            self.hiddenLayer.append(np.zeros(self.hiddenSize)) # initialize hidden layer
        
        # Initializing weights
        self.numberSynapses = inputSize*hiddenSize + hiddenSize*outputSize + (hiddenSize**2)*nbHiddenLayer
        self.W = {}
        for l in range(nbHiddenLayer+2):
            idLayer = "{0}".format(l)
            if l==0:
                I = self.inputSize
                J = self.hiddenSize
            elif l==nbHiddenLayer:
                I = self.hiddenSize
                J = self.outputSize
            else:
                I = self.hiddenSize
                J = I
            self.W.update({idLayer: np.zeros((I, J))})
 
    #########################################################
    # Initialization of network
    #########################################################

    @staticmethod
    def activationFunction(x, option='sigmoid'):
        if option=='sigmoid':
            return 1/(1+np.exp(-(x)))
        elif option=='step':
            output = np.heaviside(x, 0)
            # output[output == 0] = -1
            return  output

    def setWeights(self, vectorWeight):
        i=0
        j=-1
        for l in range(self.nbHiddenLayer+2):
            
            # Access weight of given layer
            idLayer = "{0}".format(l)
            W_layer = self.W[idLayer]
            W_size = W_layer.shape
            
            # Get the appropriate indexes of weights 
            j = j+W_size[0]*W_size[1]
            W_toSet = vectorWeight[i:j+1]
            W_toSet = W_toSet.reshape(W_size[0], W_size[1])
            i = j+1
            
            # set weights
            self.W[idLayer] = W_toSet
    
    #########################################################
    # Forward propagation
    #########################################################
    
    def run(self, X, option='step', debug=False):
        
        # Inputs
        self.inputs = X
        contributions = self.inputs.dot(self.W['0'])
        self.hiddenLayer[0] = self.activationFunction(contributions, option=option)
        if debug:
            print('contribution inputs')
            print(contributions)
            print('hidden layer 0 ------------------')
            print(self.hiddenLayer[0])
        
        # Hidden layers            
        for l in range(1, self.nbHiddenLayer):   
            idLayer = "{0}".format(l)
            contributions = self.hiddenLayer[l-1].dot(self.W[idLayer])
            self.hiddenLayer[l] = self.activationFunction(contributions, option=option)
            if debug:
                print('contribution layer')
                print(contributions)
                print('hidden layer {0} ------------------'.format(l))
                print(self.hiddenLayer[l])
                    
        # output
        if self.nbHiddenLayer==1:
            l=0
        idLayer = "{0}".format(l+1)
        contributions = self.hiddenLayer[l].dot(self.W[idLayer])
        self.outputs = self.activationFunction(contributions, option=option)
        if debug:
            print('contribution last layer')
            print(contributions)
            print('output ------------------')
            print(self.outputs)
        return self.outputs       
                  
        