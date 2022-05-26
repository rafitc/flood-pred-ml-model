import pandas as pd
import numpy as np
from keras.models import model_from_json
import warnings

json_file = open('CNNmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("CNNmodel.h5") 
print("Loaded model from disk")  
loaded_model.summary()
x_test = 
y_pred = model.predict(x_test)[:,0]

pred = np.empty((1,len(y_pred)), dtype=object)
pred = np.where(y_pred>=0.47, 1, 0)
y_train = np.reshape(Y_train,len(Y_train))
pred = np.reshape(pred,len(pred))
print(pred)