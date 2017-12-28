import numpy as np

def softmax(x):
    orig_shape = x.shape
    max_val = np.max(x, axis=-1, keepdims=True)
    x -= max_val
    x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    test3 = softmax(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,4000,12],[1,2,1,1]]))
    ans3 = np.array([[ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426],
                     [ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426],
                     [ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426],
                     [ 0.        ,  0.        ,  1.        ,  0.        ],
                     [ 0.1748777 ,  0.47536689,  0.1748777 ,  0.1748777 ]])
    assert np.allclose(test3, ans3)
    print "Done custom test"


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
