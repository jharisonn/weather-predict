# Special thanks to Dandy Naufaldi @dandynaufaldi for helping creating this project.

import pickle
import pandas as pd
import numpy as np
from keras.models import model_from_json

#Open dump file
infile = open('pca.dump','rb')
new_pca = pickle.load(infile)
infile.close()
infile = open('pcascaler.dump','rb')
new_pcascaler = pickle.load(infile)
infile.close()

#Transforming data that want to predict to scale and to PCA
predict = pd.read_csv('predict.csv',header=None)
data = predict.values
data = data.astype('float64')
data = new_pcascaler.transform(data)
data = new_pca.transform(data)

#Load weights
model = model_from_json(open('model_architecture.json').read())
model.load_weights('weights.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Predict
ans = model.predict(data)

#Print predicted values
print(ans)
for i in range(0,len(ans)):
    if np.argmax(ans[i]) == 0 :
        print("Thunderstorm")
    elif np.argmax(ans[i]) == 1:
        print("Rain")
    elif np.argmax(ans[i]) == 2:
        print("Fog")
    elif np.argmax(ans[i]) == 3:
        print("Sunshine")
