
from schema import Schema
import json
from sklearn.metrics import r2_score as R2
import numpy as np


#asdasd


#####asasdasdasd
def test_schema():
    #data normal
    with open("data/json_data.json", "r") as file:    
        test_data = json.load(file)
    
    schema = Schema([{'X': str,
                 'y': str}])
    assert schema.is_valid([test_data]) == True
    
    #data preprocess
    with open("data/json_data_preprocess.json", "r") as file:    
        test_data = json.load(file)
    
    schema = Schema([{'X': str,
                 'y': str}])
    assert schema.is_valid([test_data]) == True
    
    
    #data prediction
    with open("data/json_data_prediction.json", "r") as file:    
        test_data = json.load(file)
    
    schema = Schema([{'y_real': str,
                 'y_predict': str}])
    assert schema.is_valid([test_data]) == True
    
def test_score():
    with open("data/json_data_prediction.json", "r") as file:    
        test_data = json.load(file)
    
    y = np.array(json.loads(test_data['y_real']))
    y_ = np.array(json.loads(test_data['y_predict']))
    
    acc = np.abs(R2(y,y_))
    
    assert acc > 0.1
       

