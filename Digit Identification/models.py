import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

        #For any ML problem:
        # y=Wx + b; 
        # x:neurons
        # W:weights
        # b:bias

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        #We calculate the updated weights 'W'

        return self.w 

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        #x:neuron
        #Should I use nn.AddBias instead?
        #return nn.AddBias(self.get_weights(),x) #Why didnt this work??

        #We calculate 'Wx'

        return nn.DotProduct(x, self.get_weights())  #We are asked to use this in the question!

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # nn.as_scalar can extract a Python floating-point number from a node

        y = self.run(x) #Taking the product from above function
        product = nn.as_scalar(y) #as defined in the question!
        if product>=0:
            return 1
        else:
            return -1
        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        #As specified, I'm using the iterate_once feature 
        
        converged = False
        while not converged:
            converged = True #Then training complete
            for x,y in dataset.iterate_once(1): #as defined in the question!
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    converged = False
         

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    #Recommended hyperparameters:
    #   Hidden layer size 512
    #   Batch size 200 *For Q2 and Q3, we require that total size of the dataset be evenly divisible by the batch size.
    #   Learning rate 0.05
    #   One hidden layer (2 linear layers in total)

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #Given equation: f(x)=relu(x⋅W1​+b1​)⋅W2​+b2
        # So we need to initialize 4 parameters:
        #Usage: nn.Parameter(n, m) constructs a parameter with shape n by m
        
        self.W1 = nn.Parameter(1,512) 
        self.b1 = nn.Parameter(1,512) 
        self.W2 = nn.Parameter(512,1)
        self.b2 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #Create 2 layers

        first_layer = nn.AddBias(nn.Linear(x,self.W1),self.b1)
        #print(first_term.shape)
        first_act = nn.ReLU(first_layer) #Replacing all the neg terms
        #print(first_act.shape)
        second_layer = nn.AddBias(nn.Linear(first_act,self.W2),self.b2)
        #print(second_term.shape)
        return second_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Given: Use nn.SquareLoss as your loss.
        # Eg: loss = nn.SquareLoss(predicted_y, y)
        
        return nn.SquareLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Eg: grad_wrt_m, grad_wrt_b = nn.gradients(loss, [m, b])
        #     m.update(grad_wrt_m, multiplier)

        while True:
            gradient_W1, gradient_B1, gradient_W2, gradient_B2 = nn.gradients(self.get_loss(nn.Constant(dataset.x),nn.Constant(dataset.y)), [self.W1, self.b1, self.W2, self.b2])
            self.W1.update(gradient_W1, -0.05)
            self.b1.update(gradient_B1, -0.05)
            self.W2.update(gradient_W2, -0.05)
            self.b2.update(gradient_B2, -0.05)
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x),nn.Constant(dataset.y)))
            if loss <= 0.02:
                break             

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Each digit is of size 28 by 28 pixels, the values of which are stored in a 784-dimensional vector of floating point numbers
        # Each output we provide is a 10-dimensional vector
        # Hidden Layer = 200
        # Same as q2!

        self.W1 = nn.Parameter(784,200)
        self.b1 = nn.Parameter(1,200)
        self.W2 = nn.Parameter(200,150)
        self.b2 = nn.Parameter(1,150)
        self.W3 = nn.Parameter(150,10)
        self.b3 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # 3 layers created, same as q2!

        first_layer = nn.AddBias(nn.Linear(x,self.W1),self.b1)
        first_act = nn.ReLU(first_layer)
        second_layer = nn.AddBias(nn.Linear(first_act,self.W2),self.b2)
        second_act = nn.ReLU(second_layer)
        third_layer = nn.AddBias(nn.Linear(second_act,self.W3),self.b3)
        return third_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # You should use nn.SoftmaxLoss as your loss
        # Same as q2!
        
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Batch size = 100
        # Leaning rate = -0.5
        # Same q2!
        
        learning_rate = -0.5
        while True:
            for x,y in dataset.iterate_once(100): 
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2, grad_wrt_W3, grad_wrt_b3  = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                self.W1.update(grad_wrt_W1, learning_rate)
                self.b1.update(grad_wrt_b1, learning_rate)
                self.W2.update(grad_wrt_W2, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)
                self.W3.update(grad_wrt_W3, learning_rate)
                self.b3.update(grad_wrt_b3, learning_rate)
            accuracy = dataset.get_validation_accuracy()
            print(accuracy)
            # For 0.98, found accuracy in 25 epochs, LR = -0.05
            # For 0.98, found accuracy in 12 epochs, LR = -0.5
            # I was getting on 2/6 for positive LR!
            # For negative learning rate, I was able to run several epochs
            # But negative learning rate will maximize the loss right?
            # Then is this correct? <Have the same doubt for q2>
            if accuracy >= 0.98:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Languages: 5
        # Batch size: 100
        # LR: -0.5

        self.W1 = nn.Parameter(self.num_chars,100)
        self.b1 = nn.Parameter(1,100)
        self.W2 = nn.Parameter(100,100)
        self.b2 = nn.Parameter(1,100)
        self.W1h = nn.Parameter(100,100)
        self.b1h = nn.Parameter(1,100)
        self.W2h = nn.Parameter(100,100)
        self.b2h = nn.Parameter(1,100)
        self.Wn = nn.Parameter(100,5)
        self.bn = nn.Parameter(1,5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Same as above question!!
        
        for i in range(len(xs)):
            if i==0:
                first_layer = nn.AddBias(nn.Linear(xs[i],self.W1),self.b1)
                first_act = nn.ReLU(first_layer)
                hidden_layer = nn.AddBias(nn.Linear(first_act,self.W2),self.b2)
            else:
                first_layer = nn.AddBias(nn.Add(nn.Linear(xs[i], self.W1), nn.Linear(hidden_layer, self.W1h)),self.b1h)
                first_act = nn.ReLU(first_layer)
                second_layer = nn.AddBias(nn.Linear(first_act,self.W2h),self.b2h)
                hidden_layer = nn.ReLU(second_layer)
        return nn.AddBias(nn.Linear(hidden_layer,self.Wn),self.bn)
        

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Same as q3!

        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        accuracy = -float('inf')
        learning_rate = -0.08
        while True:
            #for i in range(20): #20 epochs
            for x,y in dataset.iterate_once(100): 
                gradients = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2, self.W1h, self.b1h, self.W2h, self.b2h, self.Wn, self.bn])
                self.W1.update(gradients[0], learning_rate)
                self.b1.update(gradients[1], learning_rate)
                self.W2.update(gradients[2], learning_rate)
                self.b2.update(gradients[3], learning_rate)
                self.W1h.update(gradients[4], learning_rate)
                self.b1h.update(gradients[5], learning_rate)
                self.W2h.update(gradients[6], learning_rate)
                self.b2h.update(gradients[7], learning_rate)
                self.Wn.update(gradients[8], learning_rate)
                self.bn.update(gradients[9], learning_rate)
            accuracy = dataset.get_validation_accuracy()
            # For LR: -0.15 Got 7/7 After 35 epochs, acc=86
            # For LR: -0.5 Got 2/7
            # For LR: -0.05 Got 7/7 After 34 epochs, acc=86
            # For LR: -0.1 Got 7/7 After 34 epochs, acc=86.4
            # For LR: -0.08 Got 7/7 After 30 epochs, acc=87.8
            if accuracy >= 0.89: 
                break
