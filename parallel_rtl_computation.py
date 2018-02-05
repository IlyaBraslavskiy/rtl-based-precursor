from rtl_func import make_grid
import multiprocessing

import numpy as np
import pandas as pd
import itertools
import time
from tqdm import tqdm
data = pd.read_pickle('work_catalog_pd')
data = data[['lat', 'long', 'class', 'l_r', 'diff_in_days_before_first_eqarthquake']]
data = data.iloc[0:1000]

def MpRtl(args):
	start = time.time()
	RTL = RTLForEachPoint(args[0],args[1])
	print time.time() - start
	return np.array(RTL)
	#return RTL


def RTLForEachPoint(r0, t0, data_frame = data):
    r, t, l, zero_r, zero_t, zero_l = [], [], [], [], [], []
    
    for idx, x in data_frame.iterrows():
        if idx != 0:
            points_before_x = (data_frame.iloc[:idx, :]).copy()
            points_before_x['time_delta'] = (x['diff_in_days_before_first_eqarthquake']\
                                             - points_before_x['diff_in_days_before_first_eqarthquake']) + 1e-3
            
            points_before_x['distance'] = 100. * np.linalg.norm(points_before_x[['lat', 'long']].values\
                                             - x[['lat', 'long']].values, 2, axis=1) + 1e-3
            
            points_rtl = (points_before_x.loc[(points_before_x['distance'] < 2.*r0)\
                                          & (points_before_x['time_delta'] < 2.*t0)]).copy()
            
            if points_rtl.shape[0] == 0:
                r.append(0.0)
                t.append(0.0)
                l.append(0.0)
            else:
                r_x = np.sum(np.exp(-1.0 * points_rtl['distance'].values / r0))
                t_x = np.sum(np.exp(-1.0 * points_rtl['time_delta'].values / t0))
                l_x = np.sum(points_rtl['l_r'].values / points_rtl['distance'].values)
                
                r.append(r_x)
                t.append(t_x)
                l.append(l_x)
    
    return np.array([0] + r), np.array([0] + t), np.array([0] + l)

r0 = [10,25,50,100]
t0 = [30,90,180,365]
grid_params = make_grid((r0,t0))

if __name__ == '__main__':
		pool = multiprocessing.Pool(4)
		answer = pool.map(MpRtl, grid_params)  
		for RTL,params in zip(answer,grid_params):
			r = RTL[0]
			t = RTL[1]
			l = RTL[2]
			r_name = "r_where:_r0=" + str(params[0]) + "_t0=" + str(params[1])	
			t_name = "t_where:_r0=" + str(params[0]) + "_t0=" + str(params[1])
			l_name = "l_where:_r0=" + str(params[0]) + "_t0=" + str(params[1])
			data[r_name], data[t_name], data[l_name] = r, t, l
		data.to_csv("test.csv", encoding='utf-8', index=False)
		print data.shape