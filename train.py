import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score as R2
# Opening JSON file
f = open('data/json_data_preprocess.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
f.close()
y = np.array(data["y"][1:-1].split(', '),dtype = float)
dsplit = data['X'][2:-2].split('], [')
X = np.array([np.array(ds.split(','),dtype = float) for ds in dsplit] )

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y)
y_ = neigh.predict(X)

if np.abs(R2(y,y_)) > 0.5:
    joblib.dump(neigh, 'joblib_model.pkl')
