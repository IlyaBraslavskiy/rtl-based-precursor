import numpy as np
import pandas as pd

def make_grid(arrays, out=None):
    """make parameter grid from given arrays

    example:

	a = [1,2,3,4,5]
	b = [1,2]
	rtl_func.make_grid((a,b))

	Out:
	array([[1, 1],
	       [1, 2],
	       [2, 1],
	       [2, 2],
	       [3, 1],
	       [3, 2],
	       [4, 1],
	       [4, 2],
	       [5, 1],
	       [5, 2]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        make_grid(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def make_anomaly_on_q_level(make_anomaly, y, quantile_up, quantile_down):
    train_size = int(make_anomaly.shape[0] * 0.25)
    
    # divide the sample to test and train
    make_anomaly_train, y_train = make_anomaly.loc[:train_size,:], y[:train_size]
    make_anomaly_test, y_test =  make_anomaly.loc[train_size:,:], y[train_size:]
    # make array for thresholds
    thresholds = []
    # iterate for each feature 
    for i in make_anomaly.columns:
        # sort feature vealues for train sample
        find_quantile = np.sort(make_anomaly_train[i].values)
        # add to thresholds array cortege (up_th, low_th) for each feature
        thresholds.append((find_quantile[int(train_size * quantile_up)], find_quantile[int(train_size * quantile_down)]))
        
    # iter for (feature, pair of th's)
    for col, thr in zip(make_anomaly.columns, thresholds):
        # change values to {0,1} comparing to quantiles
        make_anomaly[col] = (np.array(make_anomaly[col]) > 1. * thr[0]) | (np.array(make_anomaly[col]) < -1. * thr[1])
    # assign values obtained values {0,1}
    make_anomaly_train, y_train = make_anomaly.loc[:train_size-1,:], y[:train_size]
    make_anomaly_test, y_test =  make_anomaly.loc[train_size:,:], y[train_size:]
    # return happy
    return make_anomaly.astype(int)