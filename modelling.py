from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
import pandas as pd

np.random.seed(7)
data = pd.read_csv("newData.csv")

# print(data.head())
features = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5']

X_train = data.loc[:4646,features].values
Y_train = data.loc[:4646,['Weather']].values
X_test = data.loc[4646:,features].values
Y_test = data.loc[4646:,['Weather']].values

for i in range(0,len(Y_train)):
    if(Y_train[i]=="Sunshine"):
        Y_train[i] = 3
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]=="Fog"):
        Y_train[i] = 2
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]=="Rain"):
        Y_train[i] = 1
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]=="Thunderstorm"):
        Y_train[i] = 0
        Y_train[i] = int(Y_train[i])

for i in range(0,len(Y_test)):
    if(Y_test[i]=="Sunshine"):
        Y_test[i] = 3
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]=="Fog"):
        Y_test[i] = 2
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]=="Rain"):
        Y_test[i] = 1
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]=="Thunderstorm"):
        Y_test[i] = 0
        Y_test[i] = int(Y_test[i])

Y_train = Y_train.astype('int32')
Y_train = np_utils.to_categorical(Y_train,4)
Y_test = Y_test.astype('int32')
Y_test = np_utils.to_categorical(Y_test,4)

#Creating model
model = Sequential()
model.add(Dense(50, input_dim=5, init='uniform', activation='relu'))
model.add(Dense(60, init='uniform', activation='relu'))
model.add(Dense(40, init='uniform', activation='relu'))
model.add(Dense(40, init='uniform', activation='relu'))
model.add(Dense(4))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=30, batch_size=10, verbose=2, validation_data=(X_test,Y_test))
scores = model.evaluate(X_test, Y_test, verbose=0)

print("\n")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Save weights
json_string = model.to_json()
open('model_architecture.json', 'w').write(json_string)
model.save_weights('weights.h5',overwrite=True)
