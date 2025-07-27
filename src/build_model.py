def build_model(X_train, X_test,y_train, y_test, num_iterations=1000, learning_rate=0.009, print_cost=False):
    """
    Builds the logistic regression model by calling all the helper functions
    
    Arguments:
    X_train -- training data, shape (m_train, dim)
    y_train -- training labels, shape (m_train, 1)
    X_test -- test data, shape (m_test, dim)
    y_test -- test labels, shape (m_test, 1)
    num_iterations -- number of iterations for gradient descent
    learning_rate -- learning rate for gradient descent
    print_cost -- whether to print cost every 10 steps
    
    Returns:
    d -- dictionary containing information about the model
    """
    
    # Initialize parameters
    dim = X_train.shape[1]
    w, b = Initializing_zeros(dim)
    
    # Train the model
    params, grads, costs = optimize(w, b, X_train, y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    
    # Predict
    y_pred_train = predict(w, b, X_train)
    y_pred_test = predict(w, b, X_test)
    
    # Compute accuracy
    train_accuracy = 100 - np.mean(np.abs(y_pred_train - y_train.T)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_pred_test - y_test.T)) * 100

    print(f"Train accuracy: {train_accuracy:.2f} %")
    print(f"Test accuracy: {test_accuracy:.2f} %")

    d = {
        "costs": costs,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    
    return d
