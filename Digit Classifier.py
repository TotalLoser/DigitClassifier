#!/usr/bin/env python
# coding: utf-8

# # Classification of Hand-Drawn Digits
# 
# A class called `NeuralNetworkClassifier` that extends the `NeuralNetwork`.  `NeuralNetworkClassifier` will train a classifier of hand-drawn digits.
# 

# ## `NeuralNetwork` class

# In[1]:


import matplotlib.pyplot as plt


# The following code cell will write its contents to `optimizers.py` so the `import optimizers` statement in the code cell after it will work correctly.

# In[2]:


get_ipython().run_cell_magic('writefile', 'optimizers.py', "import numpy as np\n\n######################################################################\n## class Optimizers()\n######################################################################\n\nclass Optimizers():\n\n    def __init__(self, all_weights):\n        '''all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector'''\n        \n        self.all_weights = all_weights\n\n        # The following initializations are only used by adam.\n        # Only initializing m, v, beta1t and beta2t here allows multiple calls to adam to handle training\n        # with multiple subsets (batches) of training data.\n        self.mt = np.zeros_like(all_weights)\n        self.vt = np.zeros_like(all_weights)\n        self.beta1 = 0.9\n        self.beta2 = 0.999\n        self.beta1t = 1\n        self.beta2t = 1\n\n        \n    def sgd(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, verbose=True, error_convert_f=None):\n        '''\nerror_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.\ngradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error\n            with respect to each weight.\nerror_convert_f: function that converts the standardized error from error_f to original T units.\n        '''\n\n        error_trace = []\n        epochs_per_print = n_epochs // 10\n\n        for epoch in range(n_epochs):\n\n            error = error_f(*fargs)\n            grad = gradient_f(*fargs)\n\n            # Update all weights using -= to modify their values in-place.\n            self.all_weights -= learning_rate * grad\n\n            if error_convert_f:\n                error = error_convert_f(error)\n            error_trace.append(error)\n\n            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):\n                print(f'sgd: Epoch {epoch+1:d} Error={error:.5f}')\n\n        return error_trace\n\n    def adam(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, verbose=True, error_convert_f=None):\n        '''\nerror_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.\ngradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error\n            with respect to each weight.\nerror_convert_f: function that converts the standardized error from error_f to original T units.\n        '''\n\n        alpha = learning_rate  # learning rate called alpha in original paper on adam\n        epsilon = 1e-8\n        error_trace = []\n        epochs_per_print = n_epochs // 10\n\n        for epoch in range(n_epochs):\n\n            error = error_f(*fargs)\n            grad = gradient_f(*fargs)\n\n            self.mt[:] = self.beta1 * self.mt + (1 - self.beta1) * grad\n            self.vt[:] = self.beta2 * self.vt + (1 - self.beta2) * grad * grad\n            self.beta1t *= self.beta1\n            self.beta2t *= self.beta2\n\n            m_hat = self.mt / (1 - self.beta1t)\n            v_hat = self.vt / (1 - self.beta2t)\n\n            # Update all weights using -= to modify their values in-place.\n            self.all_weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)\n    \n            if error_convert_f:\n                error = error_convert_f(error)\n            error_trace.append(error)\n\n            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):\n                print(f'Adam: Epoch {epoch+1:d} Error={error:.5f}')\n\n        return error_trace\n\nif __name__ == '__main__':\n\n    import matplotlib.pyplot as plt\n    plt.ion()\n\n    def parabola(wmin):\n        return ((w - wmin) ** 2)[0]\n\n    def parabola_gradient(wmin):\n        return 2 * (w - wmin)\n\n    w = np.array([0.0])\n    optimizer = Optimizers(w)\n\n    wmin = 5\n    optimizer.sgd(parabola, parabola_gradient, [wmin],\n                  n_epochs=500, learning_rate=0.1)\n\n    print(f'sgd: Minimum of parabola is at {wmin}. Value found is {w}')\n\n    w = np.array([0.0])\n    optimizer = Optimizers(w)\n    optimizer.adam(parabola, parabola_gradient, [wmin],\n                   n_epochs=500, learning_rate=0.1)\n    \n    print(f'adam: Minimum of parabola is at {wmin}. Value found is {w}')")


# In[3]:


import numpy as np
import optimizers
import sys  # for sys.float_info.epsilon

######################################################################
## class NeuralNetwork()
######################################################################

class NeuralNetwork():


    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs, activation_function='tanh'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation_function = activation_function

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_per_layer == 0 or n_hiddens_per_layer == [] or n_hiddens_per_layer == [0]:
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        # Initialize weights, by first building list of all weight matrix shapes.
        n_in = n_inputs
        shapes = []
        for nh in self.n_hiddens_per_layer:
            shapes.append((n_in + 1, nh))
            n_in = nh
        shapes.append((n_in + 1, n_outputs))
        # self.all_weights:  vector of all weights
        # self.Ws: list of weight matrices by layer
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)
        # Define arrays to hold gradient values.
        # One array for each W array with same shape.
        self.all_gradients, self.dE_dWs = self.make_weights_and_views(shapes)

        self.trained = False
        self.total_epochs = 0
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None


    def make_weights_and_views(self, shapes):
        # vector of all weights built by horizontally stacking flatenned matrices
        # for each layer initialized with uniformly-distributed values.
        all_weights = np.hstack([np.random.uniform(size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements from vector of all weights
        # into correct shape for each layer.
        views = []
        start = 0
        for shape in shapes:
            size =shape[0] * shape[1]
            views.append(all_weights[start:start + size].reshape(shape))
            start += size
        return all_weights, views


    # Return string that shows how the constructor was called
    def __repr__(self):
        return f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_per_layer}, {self.n_outputs}, \'{self.activation_function}\')'


    # Return string that is more informative to the user about the state of this neural network.
    def __str__(self):
        result = self.__repr__()
        if len(self.error_trace) > 0:
            return self.__repr__() + f' trained for {len(self.error_trace)} epochs, final training error {self.error_trace[-1]:.4f}'


    def train(self, X, T, n_epochs, learning_rate, method='sgd', verbose=True):
        '''
train: 
  X: n_samples x n_inputs matrix of input samples, one per row
  T: n_samples x n_outputs matrix of target output values, one sample per row
  n_epochs: number of passes to take through all samples updating weights each pass
  learning_rate: factor controlling the step size of each update
  method: is either 'sgd' or 'adam'
        '''

        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            
        # Standardize X and T
        X = (X - self.Xmeans) / self.Xstds
        T = (T - self.Tmeans) / self.Tstds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = optimizers.Optimizers(self.all_weights)

        # Define function to convert value from error_f into error in original T units, 
        # but only if the network has a single output. Multiplying by self.Tstds for 
        # multiple outputs does not correctly unstandardize the error.
        if len(self.Tstds) == 1:
            error_convert_f = lambda err: (np.sqrt(err) * self.Tstds)[0] # to scalar
        else:
            error_convert_f = lambda err: np.sqrt(err)[0] # to scalar
            

        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        verbose=True,
                                        error_convert_f=error_convert_f)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         verbose=True,
                                         error_convert_f=error_convert_f)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        # Return neural network object to allow applying other methods after training.
        #  Example:    Y = nnet.train(X, T, 100, 0.01).use(X)
        return self

    def relu(self, s):
        s[s < 0] = 0
        return s

    def grad_relu(self, s):
        return (s > 0).astype(int)
    
    def forward_pass(self, X):
        '''X assumed already standardized. Output returned as standardized.'''
        self.Ys = [X]
        for W in self.Ws[:-1]:
            if self.activation_function == 'relu':
                self.Ys.append(self.relu(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
            else:
                self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        Ys = self.forward_pass(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        '''Assumes forward_pass just called with layer outputs in self.Ys.'''
        error = T - self.Ys[-1]
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if self.activation_function == 'relu':
                delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
            else:
                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
        return self.all_gradients

    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = (X - self.Xmeans) / self.Xstds
        Ys = self.forward_pass(X)
        Y = Ys[-1]
        # Unstandardize output Y before returning it
        return Y * self.Tstds + self.Tmeans


# In[4]:


X = np.arange(100).reshape((-1, 1))
T = (X - 20) ** 3 / 300000

hiddens = [10]
nnet = NeuralNetwork(X.shape[1], hiddens, T.shape[1])
nnet.train(X, T, 250, 0.01, method='adam')

plt.subplot(1, 2, 1)
plt.plot(nnet.error_trace)

plt.subplot(1, 2, 2)
plt.plot(T, label='T')
plt.plot(nnet.use(X), label='Y')
plt.legend()


# ## `NeuralNetworkClassifier` class

# In[5]:


class NeuralNetworkClassifier(NeuralNetwork):
    
    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs, activation_function='tanh'):
        super().__init__(n_inputs, n_hiddens_per_layer, n_outputs, activation_function)
    
    def train(self, X, T, n_epochs, learning_rate, method='sgd', verbose=True):
        
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  

            
        self.un = np.unique(T)
        X = (X - self.Xmeans) / self.Xstds
        #T = (T - self.Tmeans) / self.Tstds
        
        #X1 = np.hstack((np.ones((X.shape[0], 1)), X))
        TI = self.makeIndicatorVars(T)
        #self.all_weights = np.zeros(X.shape[1] * TI.shape[1])
        optimizer = optimizers.Optimizers(self.all_weights)
        to_likelihood = lambda err: np.exp(-err)
        
        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, TI], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        verbose=True,
                                        error_convert_f=to_likelihood)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, TI], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         verbose=True,
                                         error_convert_f=to_likelihood)
            

        else:
            raise Exception("method must be 'sgd' or 'adam'")
            
        self.error_trace = error_trace
        #print(self.error_trace)
        return self
    
    def makeIndicatorVars(self, T):
        if T.ndim == 1:
            T = T.reshape((-1, 1))   
       # print()
        return (T == np.unique(T)).astype(int)
    

    def softmax(self, X):
#         fixed = np.exp(X @ w)
        # fs = (X)
#         print(len(fs))
#         print(len(fs[0]))
#         fixed = []
#         for layer in fs:
#             newlayer = []
#             for e in layer:
#                 newlayer.append(e[0])
#             fixed.append(newlayer)
        fixed = np.exp(X)
        d = np.sum(fixed,axis=1).reshape((-1, 1))
        gs = fixed / d
        
#         for g in gs:
#             suum = 0
#             for e in g:
#                 suum += e
#             print(suum)
        return gs
    def f_pass(self, X):
        '''X assumed already standardized. Output returned as standardized.'''
        self.Ys = [X]
        for W in self.Ws[:-1]:
            if self.activation_function == 'relu':
                
                self.Ys.append(self.relu(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
                
            else:
                self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        w = last_W[1:, :] + last_W[0:1, :]
        #w = w.reshape()
        self.Ys.append(self.softmax(self.Ys[-1] @ w))
        return self.Ys
        
#         for W in self.Ws[:-1]:
#             self.Ys.append(self.softmax(self.Ys[-1], (W[1:, :] + W[0:1, :])))
#         last_W = self.Ws[-1]
#         self.Ys.append()
#         return self.Ys



    def error_f(self, X, T):
        
        
        
        
        Ys = self.f_pass(X)
        Y = Ys[-1]

#         print(Y)
        return - np.mean(T * np.log(Y))
    
    def gradient_f(self, X, T):
        
        error = T - self.Ys[-1]
        delta = - error
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if self.activation_function == 'relu':
                delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
            else:
                delta = delta @ self.Ws[layeri][1:, :].T #* (1 - self.Ys[layeri] ** 0)
        return self.all_gradients

            
#         #Y = self.softmax(X)
#         #print(T - Y)
#         grad = X.T @ (T - Y) 
#         #print(X.T)
#         return grad.reshape((-1))
    def use(self, X):
        X = (X - self.Xmeans) / self.Xstds
        Ys = self.f_pass(X)
        Y = Ys[-1]
        
        #print(len(Y))
        pred = []
        #un = np.unique(T)
        for entry in Y:
            ele = []
            ele.append(self.un[np.argmax(entry)])
            pred.append(ele)
        #print(pred)
        out = []
        out.append(pred)
        out.append(Y)
        return out


# Testing the classifier.  For inputs from 0 to 100, classify values less than or equal to 25 as Class Label 25, greater than 25 and less than or equal to 75 as Class Label 75, and greater than 75 as Class Label 100. 

# In[34]:


X = np.arange(100).reshape((-1, 1))
T = X.copy()
T[T <= 25] = 25
T[np.logical_and(25 < T, T <= 75)] = 75
T[T > 75] = 100
plt.plot(X, T, 'o-')
plt.xlabel('X')
plt.ylabel('Class');
#print(T)


# In[35]:


nn_class = NeuralNetworkClassifier(1, [5], 5)
result = nn_class.softmax(np.array([[-5.5, 5.5]]))
print(result)


# In[36]:


hiddens = [10]
nnet = NeuralNetworkClassifier(X.shape[1], hiddens, len(np.unique(T)))
nnet.train(X, T, 200, 0.01, method='adam', verbose=True)

plt.subplot(1, 2, 1)
plt.plot(nnet.error_trace)
plt.xlabel('Epoch')
plt.ylabel('Likelihood')

plt.subplot(1, 2, 2)
plt.plot(T + 5, 'o-', label='T + 5')  # to see, when predicted overlap T very closely
plt.plot(nnet.use(X)[0], 'o-', label='Y')
plt.legend()


# ## Now for the Hand-Drawn Digits
# 
# We will use a bunch (50,000) images of hand drawn digits from [this deeplearning.net site](http://deeplearning.net/tutorial/gettingstarted.html).  Download `mnist.pkl.gz`. 
# 
# deeplearning.net goes down a lot.  If you can't download it from there you can try getting it from [here](https://gitlab.cs.washington.edu/colinxs/neural_nets/blob/master/mnist.pkl.gz).
# 

# In[37]:


import pickle
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Xtrain = train_set[0]
Ttrain = train_set[1].reshape(-1, 1)

Xval = valid_set[0]
Tval = valid_set[1].reshape(-1, 1)

Xtest = test_set[0]
Ttest = test_set[1].reshape(-1, 1)

print(Xtrain.shape, Ttrain.shape,  Xval.shape, Tval.shape,  Xtest.shape, Ttest.shape)


# In[38]:


image0 = Xtrain[0, :]
image0 = image0.reshape(28, 28)


# In[39]:


plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(-Xtrain[i, :].reshape(28, 28), cmap='gray')
    plt.title(Ttrain[i, 0])
    plt.axis('off');


# In[40]:


classes = np.arange(10)


# In[41]:


(Ttrain == classes).sum(axis=0)


# In[42]:


(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]


# In[43]:


['Ttrain', *(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]]


# In[44]:


import pandas

result = []
result.append(['Train', *(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]])
result.append(['Tval', *(Tval == classes).sum(axis=0) / Tval.shape[0]])
result.append(['Ttest', *(Ttest == classes).sum(axis=0) / Ttest.shape[0]])
pandas.DataFrame(result)


# Time for our first experiment.  Let's train a small neural net with 5 hidden units in one layer for a small number of epochs using Adam.

# In[45]:


n_epochs = 100
learning_rate = 0.01

np.random.seed(142)

nnet = NeuralNetworkClassifier(Xtrain.shape[1], [5], len(classes))
nnet.train(Xtrain, Ttrain, n_epochs, learning_rate, method='adam', verbose=True)


# In[46]:


plt.plot(nnet.error_trace);


# In[28]:


import time

names = ('Hidden Layers', 'Train', 'Validate', 'Test', 'Time')
structs = [[5], [10], [5, 5], [10, 10], [100, 100]]
n_epochs = 500
learning_rate = 0.01
np.random.seed(142)
def percentright(Y, T):
    return 100 * np.mean(Y == T)
table = []
result = pandas.DataFrame(table, columns=names)
print(result)
for struct in structs:
    print("training for struct", struct, "...")
    entry = []
    entry.append(struct)
    Trnet = NeuralNetworkClassifier(Xtrain.shape[1], struct, len(classes))
    Vnet =  NeuralNetworkClassifier(Xval.shape[1], struct, len(classes))
    Tenet =  NeuralNetworkClassifier(Xtest.shape[1], struct, len(classes))
    
    start = time.time()
    
    Trnet.train(Xtrain, Ttrain, n_epochs, learning_rate, method='adam', verbose=True)
    Vnet.train(Xval, Tval, n_epochs, learning_rate, method='adam', verbose=True)
    Tenet.train(Xtest, Ttest, n_epochs, learning_rate, method='adam', verbose=True)
    
    elapsed = time.time() - start
    
    entry.append(percentright(Trnet.use(Xtrain)[0], Ttrain))
    entry.append(percentright(Vnet.use(Xval)[0], Tval))
    entry.append(percentright(Tenet.use(Xtest)[0], Ttest))
    
    entry.append(elapsed)
    table.append(entry)

result = pandas.DataFrame(table, columns=names)
print(result)


# In[47]:


BestNet =  NeuralNetworkClassifier(Xtest.shape[1], [10], len(classes))
BestNet.train(Xtest, Ttest, 500, 0.01, method='adam', verbose=True)


# In[48]:


classes, probs = BestNet.use(Xtest)


# After finding the best network structure, plot the nets interpretation of the data

# In[49]:


plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(-Xtest[i, :].reshape(28, 28), cmap='gray')
    plt.title(classes[i])
    plt.axis('off');


# Find the worst performing numbers to see what to improve

# In[32]:


minten = []
for j in range(10):
    minimum = (0, probs[0][Ttest[0][0]])
    for i in range(len(Ttest) - 1):
        record = (i, probs[i][Ttest[i][0]])
        if record[1] <= minimum[1] and record not in minten:
            minimum = record
    minten.append(minimum)
plt.figure(figsize=(20, 20))
i = 0
for bad in minten:
    i += 1
#for i in range(100):
    plt.subplot(10, 10, i)
    plt.imshow(-Xtest[bad[0], :].reshape(28, 28), cmap='gray')
    plt.title(classes[bad[0]][0])
    plt.axis('off');


# ### Thoughts
# depicted above are the ten images my ``BestNet`` preformed worst on, and the values above are what my network thought they were. As we can see the network seems to struggle most with zeros, especially those with a large tilt. The consistent assumption is that the digit is a 6. It's likely that the large circle in many sixes is getting mistaken here for zero.

# ## `confusion_matrix`
# 
# Returns a confusion matrix for any classification problem, returned as a `pandas.DataFrame`.  

# In[50]:


def confusion_matrix(predicted_classes, true_classes):
    
    Classes = np.unique(predicted_classes)
    #print(Classes)
    mat = []
    for ci in range(len(Classes)):
        sums = []
        for i in range(len(Classes)):
            sums.append(0)
        for i in range(len(predicted_classes)):
            if(predicted_classes[i][0] == Classes[ci]):
                sums[ci] += 1
        denom = 0
        
        for n in sums:
            denom += n
       #print(denom)
        for i in range(len(Classes)):
            sums[i] /= denom
            sums[i] *= 100
        denom = 0
        for n in sums:
            denom += n
        #print(denom)
        mat.append(sums)
    return pandas.DataFrame(mat, index=Classes, columns=Classes)


# In[51]:


Y_classes, Y_probs = nnet.use(Xtest)
confusion_matrix(Y_classes, Ttest)


# In[52]:


X = np.arange(20).reshape(20, 1)
X = np.hstack((X, X[::-1, :]))
T = np.array(['ends', 'mid'])[(np.abs(X[:, 0:1] - X[:, 1:2]) < 6).astype(int)]

np.random.seed(42)

nnet = NeuralNetworkClassifier(X.shape[1], [10, 10], len(np.unique(T)), activation_function='relu')
nnet.train(X, T, 500, 0.001, method='adam', verbose=False)

Y_classes, Y_probs = nnet.use(X)

percent_correct = 100 * np.mean(Y_classes == T)
cm = confusion_matrix(Y_classes, T)
cm


# In[ ]:




