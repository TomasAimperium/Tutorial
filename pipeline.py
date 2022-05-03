#pipeline
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
with open('data/json_data.json') as json_file:
    data = json.load(json_file)
scaler = MinMaxScaler()
pca = PCA(n_components=8)
X = np.array(json.loads(data['X'])[7000:])
y = np.array(json.loads(data['y'])[7000:])
X_ = pca.fit_transform(scaler.fit_transform(X))

d = {
    "X": str(X_.tolist()),
    "y": str(y.tolist())
}

with open('data/json_data_preprocess.json', 'w') as outfile:
    json.dump(d, outfile,indent = 4)
