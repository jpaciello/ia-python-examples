# -*- coding: utf-8 -*-
"""
@author: jpaciello
"""

# Cargar funciones de la librería de python data analysis
import pandas as pd 

# Leer csv con datos y cargar en el dataframe data
data = pd.read_csv("stockprices.csv") 

# Preview de las 5 primeras filas de data 
data.head()

# Summary stats del dataframe
dataStats = data.describe()

# Calcular variables con correlacion positiva o negativa superior a un umbral
corMatrix = data.corr(numeric_only=True)
print(corMatrix[(abs(corMatrix) > 0.75) & (corMatrix != 1)]['nasdaq'].dropna())


# Scatterplot google y nasdaq

data.plot.scatter(x='goog', y='nasdaq', c='DarkBlue')

# Cargar funciones de la librería de python sklearn y math
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

# Seleccionar variables correlacionadas
df = data[['aapl','amzn','goog','nasdaq']]

# Seleccionar x e y
x = pd.DataFrame(df[['aapl','amzn','goog']]).iloc[0:200]
y = pd.DataFrame(df['nasdaq']).iloc[0:200]
x_val = pd.DataFrame(df[['aapl','amzn','goog']]).iloc[201:]
y_val = pd.DataFrame(df['nasdaq']).iloc[201:]

# MLP: entrenar el modelo y calcular predicciones
# Documentación de parámetros: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
model = MLPRegressor(hidden_layer_sizes=(50), max_iter=150,
                  activation='relu', solver='lbfgs', random_state=1)
model.fit(x, y.values.ravel())
y_pred = model.predict(x_val)

# Line plot predicted vs real values
d = pd.DataFrame({'y_val':y_val['nasdaq'].tolist(),'y_pred':y_pred.tolist()})
d.plot.line(y=['y_pred','y_val'])

# calcular coeficiente r2 y pearson
#print("Coeficiente r2: " + str(model.score(x, y)))
#print("Coeficiente de Pearson (r): " + str(sqrt(model.score(x, y))))

# calcular root mean square error (RMSE) y mean absolute percetage error (MAPE)
print("Raíz del Error cuadrático Medio (RMSE): " + str(root_mean_squared_error(y_val, y_pred)))
#print("Media de Nasdaq: " + str(y_val['nasdaq'].mean()))
print("Mape (%): " + str(mean_absolute_percentage_error(y_val, y_pred)))

######### PCA
from sklearn.decomposition import PCA

# Crear PCA que explica el 98% de varianza acumulada
pca = PCA(n_components=0.98, svd_solver='full')

# Training/validation por holdout
x = data.iloc[0:200,1:28]
x_val = data.iloc[200:,1:28]
y = data.iloc[0:200,28:29]
y_val = data.iloc[200:,28:29]

# Aplicar PCA
x = pca.fit_transform(x)

# Imprimir componentes y varianza acumulada
print(x.shape)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
print(pca.components_)

# Transformar x_val
x_val = pca.transform(x_val)

# entrenar el modelo y calcular predicciones
model = MLPRegressor(hidden_layer_sizes=(7), max_iter=2000,
                  activation='relu', solver='lbfgs', random_state=1)
model.fit(x, y.values.ravel())
y_pred = model.predict(x_val)

# calcular coeficiente r2 y pearson
#print("Coeficiente r2: " + str(model.score(x, y)))
#print("Coeficiente de Pearson (r): " + str(sqrt(model.score(x, y))))

# calcular root mean square error (RMSE) y mean absolute percetage error (MAPE)
print("Raíz del Error cuadrático Medio (RMSE): " + str(root_mean_squared_error(y_val, y_pred)))
#print("Media de Nasdaq: " + str(y_val['nasdaq'].mean()))
print("Mape (%): " + str(mean_absolute_percentage_error(y_val, y_pred)))

# Line plot predicted vs real values
d = pd.DataFrame({'y_val':y_val['nasdaq'].tolist(),'y_pred':y_pred.tolist()})
d.plot.line(y=['y_pred','y_val'])
