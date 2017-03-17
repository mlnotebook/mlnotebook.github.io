import numpy as np

# Transfer function
def sigmoid(x, Derivative=False):
    if not Derivative:
        return 1 / (1 + np.exp (-x))
    else:
        out = sigmoid(x)
        return out * (1 - out)
    
def linear(x, Derivative=False):
    if not Derivative:
        return x
    else:
        return np.ones(x.shape)
    
def gaussian(x, Derivative=False):
    if not Derivative:
        return np.exp(-x**2)
    else:
        return -2 * x * np.exp(-x**2)
    
def tanh(x, Derivative=False):
    if not Derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2
		
class backPropNN:
    """Class defining a NN using Back Propagation"""
    
    # Class Members (internal variables that are accessed with backPropNN.member)
    
    numLayers = 0
    shape = None
    weights = []
    
    # Class Methods (internal functions that can be called)
    
    def __init__(self, numNodes, transferFunctions=None):
        """Initialise the NN - setup the layers and initial weights"""

        # Layer info
        self.numLayers = len(numNodes) - 1
        self.shape = numNodes
        
        if transferFunctions is None:
            layerTFs = []
            for i in range(self.numLayers):
                if i == self.numLayers - 1:
                    layerTFs.append(linear)
                else:
                    layerTFs.append(sigmoid)
        else:
            if len(numNodes) != len(transferFunctions):
                raise ValueError("Number of transfer functions must match the number of layers: minus input layer")
            elif transferFunctions[0] is not None:
                raise ValueError("The Input layer doesn't need a a transfer function: give it [None,...]")
            else:
                layerTFs = transferFunctions[1:]

        self.tFunctions = layerTFs        
        
        
        # Input/Output data from last run
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []     
       
        # Create the weight arrays
        for (l1,l2) in zip(numNodes[:-1],numNodes[1:]):
            self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))       
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))
        
        
    # Forward Pass method
    """Get the input data and run it through the NN"""
    def FP(self, input):

        delta = []
        numExamples = input.shape[0]

        # Clean away the values from the previous layer
        self._layerInput = []
        self._layerOutput = []
        
        for index in range(self.numLayers):
            #Get input to the layer
            if index ==0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, numExamples])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,numExamples])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFunctions[index](layerInput))
            
        return self._layerOutput[-1].T
            
    # backPropagation method
    def backProp(self, input, target, learningRate = 0.2, momentum=0.5):
        """Get the error, deltas and back propagate to update the weights"""

        delta = []
        numExamples = input.shape[0]
        
        # First run the network
        self.FP(input)
                 
        # Calculate the deltas for each node
        for index in reversed(range(self.numLayers)):
            if index == self.numLayers - 1:
                # If the output layer, then compare to the target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.tFunctions[index](self._layerInput[index], True))
            else:
                # If a hidden layer. compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:] * self.tFunctions[index](self._layerInput[index], True))
                
        # Compute updates to each weight
        for index in range(self.numLayers):
            delta_index = self.numLayers - 1 - index
            
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, numExamples])])
            else:
                layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1,self._layerOutput[index -1].shape[1]])])

            thisWeightDelta = np.sum(\
                    layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0) \
                    , axis = 0)
            
            weightDelta = learningRate * thisWeightDelta + momentum * self._previousWeightDelta[index]
            
            self.weights[index] -= weightDelta
            
            self._previousWeightDelta[index] = weightDelta
            
        return error
		
Input = np.array([[0,0],[1,1],[0,1],[1,0]])
Target = np.array([[0.0],[0.0],[1.0],[1.0]])
transferFunctions = [None, sigmoid, linear]
    
NN = backPropNN((2,2,1), transferFunctions)

maxIterations = 100000
minError = 1e-5

for i in range(maxIterations + 1):
    Error = NN.backProp(Input, Target, learningRate=0.2, momentum=0.5)
    if i % 2500 == 0:
        print("Iteration {0}\tError: {1:0.6f}".format(i,Error))
    if Error <= minError:
        print("Minimum error reached at iteration {0}".format(i))
        break

Output = NN.FP(Input)

print 'Input \tOutput \t\tTarget'
for i in range(Input.shape[0]):
    print '{0}\t {1} \t{2}'.format(Input[i], Output[i], Target[i])