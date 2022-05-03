#predict
import json
import numpy as np
import joblib
with open('data/json_data_preprocess.json') as json_file:
    data = json.load(json_file)

X = np.array(json.loads(data['X']))
y = np.array(json.loads(data['y']))

loaded_model = joblib.load('model/joblib_model.pkl')
result = loaded_model.predict(X)

d = {
    "y_real": str(y.tolist()),
    "y_predict": str(result.tolist())
}


with open('data/json_data_prediction.json', 'w') as outfile:
    json.dump(d, outfile,indent = 4)
