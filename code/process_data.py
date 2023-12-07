
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
import pickle
import pandas as pd

path_datos_samuel = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto3'
path_datos_juan = '/Users/juandramirezj/Documents/Universidad_MIIND/ACTD/proyecto_final/data'
path_datos_actual = path_datos_juan
# Cargar los datos
#data = pd.read_csv(path_datos_actual+'/DatosFiltradosSaber11.csv', delimiter=",")

#print (data.head())
#print(data)

with open(path_datos_actual+'/DatosFiltradosSaber11.csv', 'r', encoding='utf-8') as file:
    content = file.read()

# Divide el contenido en filas basadas en el salto de línea
rows = content.split('\n')

# Divide cada fila en columnas basadas en el punto y coma
data_list = [row.split(',') for row in rows]

# Construye el DataFrame
data = pd.DataFrame(data_list)

print(data)
data.head()
# Eliminar comillas dobles en todas las celdas del DataFrame
data = data.drop(len(data)-1, axis=0)
data = data.applymap(lambda x: x.strip('"'))
data = data.replace('', np.nan)
# Muestra las primeras filas del DataFrame después de eliminar las comillas
print(data.head())

# Get NA summary
nan_summary = data.isna().sum()
print(nan_summary)

nan_summary = nan_summary[nan_summary > 0]
print(nan_summary)

# Establece la primera fila como encabezados
data.columns = data.iloc[0]

# Elimina la primera fila del DataFrame
data = data[1:]

# Reinicia los índices del DataFrame después de la eliminación
data = data.reset_index(drop=True)

data['PUNT_INGLES'] = data['PUNT_INGLES'].astype(float)
data['PUNT_MATEMATICAS'] = data['PUNT_MATEMATICAS'].astype(float)
data['PUNT_SOCIALES_CIUDADANAS'] = data['PUNT_SOCIALES_CIUDADANAS'].astype(float)
data['PUNT_C_NATURALES'] = data['PUNT_C_NATURALES'].astype(float)
data['PUNT_LECTURA_CRITICA'] = data['PUNT_LECTURA_CRITICA'].astype(float)
data['PUNT_GLOBAL'] = data['PUNT_GLOBAL'].astype(float)
for col in data.select_dtypes(include=['float64', 'int64']):
    data[col].fillna(data[col].mean(), inplace=True)

# For categorical columns, fill NaN with mode
for col in data.select_dtypes(include=['object']):
    data[col].fillna(data[col].mode()[0], inplace=True)

data_numeric =  data.copy()


# Assuming df is your DataFrame and 'column_name' is the name of your column
data['PUNT_INGLES'] = pd.qcut(data['PUNT_INGLES'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['PUNT_MATEMATICAS'] = pd.qcut(data['PUNT_MATEMATICAS'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['PUNT_SOCIALES_CIUDADANAS'] = pd.qcut(data['PUNT_SOCIALES_CIUDADANAS'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['PUNT_C_NATURALES'] = pd.qcut(data['PUNT_C_NATURALES'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['PUNT_LECTURA_CRITICA'] = pd.qcut(data['PUNT_LECTURA_CRITICA'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['PUNT_GLOBAL'] = pd.qcut(data['PUNT_GLOBAL'], q=2, labels=["Q1", "Q2"])

with open(path_datos_actual+'/filtered_data.pkl', 'wb') as file:
    pickle.dump(data, file)
with open(path_datos_actual+'/filtered_data_numeric.pkl', 'wb') as file:
    pickle.dump(data_numeric, file)