#Full dataset of x-rays used can be found online at NIH website

import numpy as np

def load_example_by_name(vol_name):

    X = np.load(vol_name)['image2D']
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]
    return tuple(return_vals)


def example_gen(vol_names):
    #idx = 0
    while(True):
        idx = np.random.randint(len(vol_names))
        X = np.load(vol_names[idx])['image2D']
        X = np.reshape(X, (1,) + X.shape + (1,))

        return_vals = [X]

        yield tuple(return_vals)
