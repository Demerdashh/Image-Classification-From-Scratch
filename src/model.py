# Logistic Regression sigmoid function
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/ (1+np.exp(-z))

    return s


# Initializing whights and bias
def Initializing_zeros(dim):
    """
    This function creates a vector of zeros of shape (1,dim) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (1,dim)
    b -- initialized scalar (corresponds to the bias) of type float
    """

    w = np.zeros((1,dim)) # shape (1, dim) to ensure column vector (avoid rank-1 array)
    b = 0.0
    return w, b


# forward and backward propagation
def propagate(w, b, X, y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (1, dim)
    b -- bias, a scalar
    X -- data of size (number of examples, dim)
    Y -- true "label" vector (containing 0 if cat, 1 if dog) of size (number of examples, 1)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dldw -- gradient of the loss with respect to w, thus same shape as w)
            (dldb -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    """
    m = X.shape[0] # number of training examples
    
    # forward propagation
    yhat = sigmoid(np.dot(X, w.T) + b)          # shape: (m,1)

    cost = -(1/m)*(np.sum(y*np.log(yhat)+(1-y)*np.log(1-yhat)))
    cost = np.squeeze(np.array(cost))
    # backward propagation
    dldz = yhat - y                          #shape: (m, 1)
    dldw = (1/m)*np.dot(X.T,dldz)            #shape: (dim, 1)
    dldb = (1/m)*np.sum(yhat - y)            #scalar

    dldw = dldw.T    # to match w's shape to update it safely

    grads = {"dldw":dldw,
             "dldb":dldb}
    
    return grads, cost



def optimize(w, b, X, y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (1, dim)
    b -- bias, a scalar
    X -- data of shape (number of examples, dim)
    y -- true "label" vector (containing 0 if cat, 1 if dog), of shape (number of examples, 1)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 10 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    costs = []
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, y)

        dldw = grads['dldw']
        dldb = grads['dldb']
        # Updating parameters
        w = w -learning_rate*dldw
        b = b -learning_rate*dldb

        # Recording the cost every 10 iterations
        if i % 10 == 0:
            costs.append(cost)
            print ("Cost after iteration %i: %f" %(i, cost))
    #end of loop       


    params = {'w':w,
              'b':b}

    grads = {'dldw':dldw,
             'dldb':dldb}

    return params, grads, costs



def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (1,dim)
    b -- bias, a scalar
    X -- data of size (number of examples, dim)
    
    Returns:
    y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[0]
    y_pred= np.zeros((1, m))
    yhat = sigmoid(np.dot(w, X.T)+b) #shape: (1,m)

    for i in range(yhat.shape[1]):
        if yhat[0, i] > 0.5:
            y_pred[0, i] = 1
        else:
            y_pred[0, i] = 0
    #end of loop

    return y_pred

