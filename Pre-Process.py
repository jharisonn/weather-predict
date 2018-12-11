# Special thanks to Dandy Naufaldi @dandynaufaldi for helping creating this project.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

url = "data.csv"

df = pd.read_csv(url,
names=['Max Temperature','Min Temperature','Max DewPoint','Min DewPoint','Max Humidity','Min Humidity','Max Pressure','Min Pressure','Max Visibility','Min Visibility','Mean Wind Speed','Weather'])


features = ['Max Temperature','Min Temperature','Max DewPoint','Min DewPoint','Max Humidity','Min Humidity','Max Pressure','Min Pressure','Max Visibility','Min Visibility','Mean Wind Speed']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Weather']].values
# Standardizing the featufres
scaler = StandardScaler()
x = scaler.fit_transform(x)
# x = StandardScaler().fit_transform(x)
filename = 'pcascaler.dump'
outfile = open(filename,'wb')

#Creating dump for Scaler
pickle.dump(scaler,outfile)
outfile.close()

# Transform from 11 dimensions to 5 dimensions
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])

# Concatenate Weather column to 5 PCA Dimensions
finalDf = pd.concat([principalDf, df[['Weather']]],axis=1)
print(finalDf)
finalDf.to_csv("newData.csv")

# Creating dump file for PCA
filename = 'pca.dump'
outfile = open(filename,'wb')

pickle.dump(pca,outfile)
outfile.close()
