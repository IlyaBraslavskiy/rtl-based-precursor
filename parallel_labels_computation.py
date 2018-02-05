from rtl_func import make_grid
import multiprocessing

import numpy as np
import pandas as pd
import itertools
import time

data = pd.read_pickle('work_catalog_pd')
data = data[['lat', 'long', 'class', 'l_r', 'diff_in_days_before_first_eqarthquake']]
#data = data.iloc[0:50000]

def MPlabels(args):
	start = time.time()
	label = ByCoordXYTSayIsItBigBangEnoughNear(args[0],args[1],args[2],args[3])
	print time.time() - start
	return label


def ByCoordXYTSayIsItBigBangEnoughNear(m_c, r_c, delta_c, t_c,idx_features = data):
    alarms = []
    # consider earthquakes with magnitudes m > m_c
    idx_magnitude = idx_features.loc[idx_features['class'] > m_c]
    for _, x in idx_features.iterrows():
        # for all events compute distance to earthquake-events
        distance = np.array(100. * np.linalg.norm(idx_magnitude[['lat', 'long']].values - x[['lat', 'long']].values,\
                                                  2, axis=1))
        # same with time distance
        time_distance = np.array(idx_magnitude['diff_in_days_before_first_eqarthquake'].values \
                         - x['diff_in_days_before_first_eqarthquake'])
        
        # if event satisfies conditions, then assigned value 1
        is_alarm = int(np.sum((distance < r_c) * (time_distance < t_c) * (time_distance > delta_c)) > 0)
        alarms.append(is_alarm)
    return alarms

m_c = [5.,6.,7.]
r_c = [50.,100.,200.]
delta_c = [10.,30.,90.]
t_c = [180.,365.]

grid_params = make_grid((m_c,r_c,delta_c,t_c))
X = pd.DataFrame()

if __name__ == '__main__':
		pool = multiprocessing.Pool(4)
		answer = pool.map(MPlabels, grid_params)  
		for label,params in zip(answer,grid_params):
			index_name = "index_where:_m_c=" + str(params[0]) + "_r_c=" + str(params[1]) + "_delta_c=" + str(params[2]) + "_t_c" + str(params[3])
			X[index_name] = label
		X.to_csv("labels.csv", encoding='utf-8', index=False)
		print X.shape
