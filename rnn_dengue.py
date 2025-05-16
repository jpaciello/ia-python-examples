# -*- coding: utf-8 -*-
"""
@author: jpaciello
"""

# Cargar funciones de la librería de python data analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Leer csv con datos y cargar en el dataframe data
data = pd.read_csv("dengue_clima.csv") 

#Obtener variables con correlación en abs mayor o igual a 0.3
cm = data.corr()
print(cm[(abs(cm['cantidad'])>=0.3)].cantidad)


#Feature selection en base a correlación, evitando autoregressive model

df = data[['temperatura_media_media(-7)','temperatura_media_media(-8)','temperatura_media_media(-9)',
           'temperatura_media_media(-10)','temperatura_media_media(-11)','nivel(-5)','nivel(-6)',
           'nivel(-7)','nivel(-8)','nivel(-9)','nivel(-10)','nivel(-11)', 'cantidad']]

# Normalizacion
sc = MinMaxScaler(feature_range = (0, 1))
sc_y = MinMaxScaler(feature_range = (0, 1))

# 276 registros para train, 52 para validacion (52 semanas equivalente a 1 año)
# última columna es el target, todas las previas son input
col = df.shape[1]
n = 276
X = sc.fit_transform(df.iloc[:,0:col-1])
Y = sc_y.fit_transform(df.iloc[:,col-1:col])
X_train = X[0:n,]
X_test = X[n:,]
y_train = Y[0:n,]
y_test = Y[n:,]

# Reshaping train and validation - Tensorflow (3-d)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = sc_y.inverse_transform(y_test)

# importar keras 
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.layers import LSTM
from keras.api.layers import Dropout

# Initializing the RNN
regressor = Sequential()
units = 5
dropout = 0.25
activation = 'tanh'

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = units, return_sequences = True, input_shape = (X_train.shape[1], 1), activation=activation))
regressor.add(Dropout(dropout))

# Adding a second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = units, return_sequences = True, activation=activation))
regressor.add(Dropout(dropout))

# Adding a third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = units, activation=activation))
regressor.add(Dropout(dropout))

# Adding the output layer
regressor.add(Dense(units = 1, activation=activation))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 10)

# Prediction
y_pred = regressor.predict(X_test)
y_pred[y_pred < 0] = 0
y_pred = sc_y.inverse_transform(y_pred)

# RMSE
from sklearn.metrics import root_mean_squared_error
print("Raíz del Error Cuadrático Medio (RMSE): " + str(root_mean_squared_error(y_test, y_pred)))
print("Average Y (AVG): " + str(y_test.mean()))
print("Standard Deviation Y (SD): " + str(y_test.std()))

# Graficar resultado
plt.plot(y_test, color = 'red', label = 'Cantidad Real')
plt.plot(y_pred, color = 'blue', label = 'Prediccion')
plt.title('Prediccion de cantidad de casos de dengue')
plt.xlabel('Semana')
plt.ylabel('Cantidad')
plt.legend()
plt.show()
