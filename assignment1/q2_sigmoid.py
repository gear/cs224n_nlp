#!/usr/bin/env python

import numpy as np

def _sigmoid(scalar_x):
    if scalar_x > 0:
        return 1.0 / (1 + np.exp(-scalar_x))
    else:
        return np.exp(scalar_x) / (1.0 + np.exp(scalar_x))

def sigmoid(x):
    f = np.vectorize(_sigmoid) 
    s = f(x)
    return s


def sigmoid_grad(s):
    ds = s * (1 - s)
    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "You should verify these results by hand!\n"


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."


if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
