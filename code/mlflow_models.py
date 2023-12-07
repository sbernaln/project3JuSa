from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

path_datos_samuel = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto3'
path_datos_juan = '/Users/juandramirezj/Documents/Universidad - MIIND/ACTD/proyecto_1/project_1_ACTD/data'
path_datos_actual = path_datos_samuel

with open(path_datos_actual+'/DatosFiltradosSaber11.csv', 'r', encoding='utf-8') as file:
    content = file.read()

# Divide el contenido en filas basadas en el salto de línea
rows = content.split('\n')

# Divide cada fila en columnas basadas en el punto y coma
data_list = [row.split(',') for row in rows]

# Construye el DataFrame
data = pd.DataFrame(data_list)


print(data)

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

data['PUNT_GLOBAL'] = pd.qcut(data['PUNT_GLOBAL'], q=2, labels=["Q1", "Q2"])

# Assuming df is your DataFrame and 'column_name' is the name of your column


print(data)

mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("models_Pro3")

model1 = joblib.load(path_datos_actual + '/model1.pkl')
model2 = joblib.load(path_datos_actual + '/model2.pkl')
model3 = joblib.load(path_datos_actual + '/model3.pkl')

train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

inference1 = VariableElimination(model1)
# For each row in the test_data, predict the probability of "lung"
target_probabilities1 = []
for index, row in test_data.iterrows():
    prob = inference1.query(variables=["PUNT_GLOBAL"], evidence={"COLE_JORNADA": row["COLE_JORNADA"],
                                                                  "COLE_CALENDARIO": row["COLE_CALENDARIO"],
                                                                  "COLE_BILINGUE": row["COLE_BILINGUE"]})
    target_probabilities1.append(prob)

inference2 = VariableElimination(model2)
# For each row in the test_data, predict the probability of "lung"
target_probabilities2 = []
for index, row in test_data.iterrows():
    prob = inference2.query(variables=["PUNT_GLOBAL"], evidence={"FAMI_TIENECOMPUTADOR": row["FAMI_TIENECOMPUTADOR"],
                                                                  "FAMI_TIENEINTERNET": row["FAMI_TIENEINTERNET"],
                                                                  "FAMI_TIENELAVADORA": row["FAMI_TIENELAVADORA"]})
    target_probabilities2.append(prob)

inference3 = VariableElimination(model3)
# For each row in the test_data, predict the probability of "lung"
target_probabilities3 = []
for index, row in test_data.iterrows():
    prob = inference3.query(variables=["PUNT_GLOBAL"], evidence={"FAMI_EDUCACIONMADRE": row["FAMI_EDUCACIONMADRE"],
                                                                  "FAMI_EDUCACIONPADRE": row["FAMI_EDUCACIONPADRE"],
                                                                  "FAMI_ESTRATOVIVIENDA": row["FAMI_ESTRATOVIVIENDA"],
                                                                  "FAMI_TIENECOMPUTADOR": row["FAMI_TIENECOMPUTADOR"],
                                                                  "FAMI_TIENEINTERNET": row["FAMI_TIENEINTERNET"]})
    target_probabilities3.append(prob)

def get_probs_dropout(probs_vector):
    probs_final=[]
    for prob in probs_vector:
        value_prob=prob.values[1]
        probs_final.append(value_prob)

    return probs_final



test_data["PUNT_GLOBAL_TARGET"] = test_data["PUNT_GLOBAL"].apply(lambda x: 1 if x == "Q2" else 0)
train_data["PUNT_GLOBAL_TARGET"] = train_data["PUNT_GLOBAL"].apply(lambda x: 1 if x == "Q2" else 0)
def plot_roc(probs, true_labels, title="ROC Curve", figsize=(10, 6)):
    """
    This function plots a ROC curve given predicted probabilities and actual labels.
    
    :param probs: Array-like, predicted probabilities.
    :param true_labels: Array-like, actual target values.
    :param title: String, desired title of the plot.
    :param figsize: tuple, size of the figure.
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    
    # Calculate AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    
    # Set aesthetics
    sns.set_style("whitegrid")
    
    # Create the plot
    plt.figure(figsize=figsize)
    lw = 2  # Line width
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=13)
    
    # Display the plot
    plt.show()


def evaluate_performance(predictions_prob, true_labels, threshold=0.5):
    # Convert the probabilities into predictions based on the specified threshold
    predictions = [1 if prob >= threshold else 0 for prob in predictions_prob]

    # Compute TP, FP, TN, and FN
    TP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 1])
    FP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 0])
    TN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 0])
    FN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 1])

    # Compute the metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "True positives": TP,
        "False positives": FP,
        "True negatives": TN,
        "False negatives": FN
    }

probs_dropout1 = get_probs_dropout(target_probabilities1)
probs_dropout2 = get_probs_dropout(target_probabilities2)
probs_dropout3 = get_probs_dropout(target_probabilities3)

series = pd.Series(probs_dropout3)
# Compute the mean of the non-NaN values
mean_value = series.mean()

# Replace NaN values with the mean
series.fillna(mean_value, inplace=True)

# Convert the series back to a list, if needed
new_probs_3 = series.tolist()


probs_df_1 = pd.DataFrame(probs_dropout1, columns=['Probability'])
probs_df_2 = pd.DataFrame(probs_dropout2, columns=['Probability'])
probs_df_3 = pd.DataFrame(probs_dropout3, columns=['Probability'])


train_data['PUNT_GLOBAL_TARGET'] = train_data['PUNT_GLOBAL_TARGET'].astype(float)
quantile_cutoff = 1 - train_data['PUNT_GLOBAL_TARGET'].mean()
print(quantile_cutoff)


cutoff1 = probs_df_1['Probability'].quantile(quantile_cutoff)
cutoff2 = probs_df_2['Probability'].quantile(quantile_cutoff)
cutoff3 = probs_df_3['Probability'].quantile(quantile_cutoff)

performance1 = evaluate_performance(probs_dropout1, test_data['PUNT_GLOBAL_TARGET'], threshold=cutoff1)
performance2 = evaluate_performance(probs_dropout2, test_data['PUNT_GLOBAL_TARGET'], threshold=cutoff2)
performance3 = evaluate_performance(new_probs_3, test_data['PUNT_GLOBAL_TARGET'], threshold=cutoff3)

with mlflow.start_run(experiment_id=experiment.experiment_id):
    mlflow.sklearn.log_model(model1, "model1")
    mlflow.sklearn.log_model(model2, "model2")
    mlflow.sklearn.log_model(model3, "model3")

accuracy_m1=performance1['Accuracy']
precision_m1= performance1['Precision']
recall_m1 = performance1['Recall']
specificity_m1 = performance1['Specificity']
true_positives_m1 = performance1['True positives']
false_positives_m1 = performance1['False positives']
true_negatives_m1 = performance1['True negatives']
false_negatives_m1 = performance1['False negatives']

accuracy_m2=performance2['Accuracy']
precision_m2= performance2['Precision']
recall_m2 = performance2['Recall']
specificity_m2 = performance2['Specificity']
true_positives_m2 = performance2['True positives']
false_positives_m2 = performance2['False positives']
true_negatives_m2 = performance2['True negatives']
false_negatives_m2 = performance2['False negatives']

accuracy_m3=performance3['Accuracy']
precision_m3= performance3['Precision']
recall_m3 = performance3['Recall']
specificity_m3 = performance3['Specificity']
true_positives_m3 = performance3['True positives']
false_positives_m3 = performance3['False positives']
true_negatives_m3 = performance3['True negatives']
false_negatives_m3 = performance3['False negatives']

mlflow.log_metric("accuracy", accuracy_m1)
mlflow.log_metric("precision", precision_m1)
mlflow.log_metric("recall", recall_m1)
mlflow.log_metric("specificity", specificity_m1)
mlflow.log_metric("true_positives", true_positives_m1)
mlflow.log_metric("false_positives", false_positives_m1)
mlflow.log_metric("true_negatives", true_negatives_m1)
mlflow.log_metric("false_negatives", false_negatives_m1)

mlflow.log_metric("accuracy", accuracy_m2)
mlflow.log_metric("precision", precision_m2)
mlflow.log_metric("recall", recall_m2)
mlflow.log_metric("specificity", specificity_m2)
mlflow.log_metric("true_positives", true_positives_m2)
mlflow.log_metric("false_positives", false_positives_m2)
mlflow.log_metric("true_negatives", true_negatives_m2)
mlflow.log_metric("false_negatives", false_negatives_m2)

mlflow.log_metric("accuracy", accuracy_m3)
mlflow.log_metric("precision", precision_m3)
mlflow.log_metric("recall", recall_m3)
mlflow.log_metric("specificity", specificity_m3)
mlflow.log_metric("true_positives", true_positives_m3)
mlflow.log_metric("false_positives", false_positives_m3)
mlflow.log_metric("true_negatives", true_negatives_m3)
mlflow.log_metric("false_negatives", false_negatives_m3)



plot_roc(probs_dropout1, test_data['PUNT_GLOBAL_TARGET'], title="Model 1 ROC Curve: Testing data")
plot_roc(probs_dropout2, test_data['PUNT_GLOBAL_TARGET'], title="Model 2 ROC Curve: Testing data")
plot_roc(new_probs_3, test_data['PUNT_GLOBAL_TARGET'], title="Model 3 ROC Curve: Testing data")

