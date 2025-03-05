"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor

def test_prediction():
    """Test that the logistic function correctly returns probabilities between 0 and 1."""
    
    model = LogisticRegressor(num_feats=2)  #small test model
    
    #manually set weights to zero (ensures sigmoid(0) = 0.5)
    model.W = np.array([0.0, 0.0, 0.0])  # 2 features + bias term
    
    #create a simple test input (3 samples, 2 features each)
    X_test = np.array([[1, 2], [-1, -2], [0, 0]])
    
    #add bias term (since train_model adds a bias column)
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    
    #get predictions
    y_pred = model.make_prediction(X_test)
    
    #check that predictions are between 0 and 1
    assert np.all(y_pred >= 0) and np.all(y_pred <= 1), "Predictions are not in range [0,1]"

    #check sigmoid behavior: Input [0,0] should return 0.5
    assert np.isclose(y_pred[2], 0.5, atol=1e-5), f"Expected 0.5, got {y_pred[2]}"


def test_loss_function():
	"""Test that binary cross-entropy loss is computed correctly."""
    
	model = LogisticRegressor(num_feats=1)
    
    #simple test case: 2 samples with known probabilities
	y_true = np.array([1, 0])  #true labels
	y_pred = np.array([0.9, 0.1])  #model predictions
    
    #compute loss
	loss = model.loss_function(y_true, y_pred)
    
    #expected loss using loss formula
	expected_loss = -np.mean([np.log(0.9), np.log(0.9)])  # Should be close to 0.105
     
    #allow slight numerical errors cus bits aren't too precise
	assert np.isclose(loss, expected_loss, atol=1e-3), f"Expected {expected_loss}, got {loss}"

def test_gradient():
    """Test that the gradient calculation is correct."""
    
    model = LogisticRegressor(num_feats=2)
    
    #simple input: 2 samples, 2 features each
    X_test = np.array([[1, 2], [-1, -2]])  
    y_true = np.array([1, 0])  # True labels

    #add bias term
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    
    #compute predictions (sigmoid(XW), where W is initialized randomly)
    y_pred = model.make_prediction(X_test)

    #compute expected gradient manually: (1/N) * X^T (y_pred - y_true)
    error = y_pred - y_true
    expected_gradient = np.dot(X_test.T, error) / X_test.shape[0]
    
    #compute gradient from model
    gradient = model.calculate_gradient(y_true, X_test)
    
    #check if gradients are close
    assert np.allclose(gradient, expected_gradient, atol=1e-3), "Gradient does not match expected values."


def test_training():
    """Test that training updates weights."""
    
    model = LogisticRegressor(num_feats=2, learning_rate=0.1, max_iter=5)
    
    #simple dataset: 4 samples, 2 features each
    X_train = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])  
    y_train = np.array([1, 0, 0, 1])  # True labels
    
    #track initial weights
    initial_W = model.W.copy()
    
    #train model (train_model will add bias term internally)
    model.train_model(X_train, y_train, X_train, y_train)
    
    #ensure weights have changed
    assert not np.allclose(initial_W, model.W), "Weights did not update during training"
    
    #check that loss decreased
    assert model.loss_hist_train[-1] < model.loss_hist_train[0], "Training loss did not decrease"

