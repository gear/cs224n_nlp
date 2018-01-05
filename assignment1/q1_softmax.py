import numpy as np

def softmax(x):
    orig_shape = x.shape
    max_val = np.max(x, axis=-1, keepdims=True)
    x -= max_val
    x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    assert x.shape == orig_shape
    return x


def test_softmax_basic():
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
    print "Running your tests..."
<<<<<<< HEAD
    test3 = softmax(np.array([[1,2,3,4,5], [1000, 2000, 3000, 5, 7]]))
    ans3 = np.array([
        [0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865],
        [0., 0., 1., 0., 0.]])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)
=======
    test3 = softmax(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,4000,12],[1,2,1,1]]))
    ans3 = np.array([[ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426],
                     [ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426],
                     [ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426],
                     [ 0.        ,  0.        ,  1.        ,  0.        ],
                     [ 0.1748777 ,  0.47536689,  0.1748777 ,  0.1748777 ]])
    assert np.allclose(test3, ans3)
    print "Done custom test"
>>>>>>> 52c218d18756d5c4e9f3f3c6369e742ce51d9421


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
