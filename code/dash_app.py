

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
from dash_core_components import Tabs, Tab
 


path_modelo = '/home/ubuntu/'  
path_data = '/home/ubuntu/' 


path_modelo = '/Users/juandramirezj/Documents/Universidad_MIIND/ACTD/proyecto_final/project3JuSa/models'
path_data = '/Users/juandramirezj/Documents/Universidad_MIIND/ACTD/proyecto_final/project3JuSa/data'


with open(path_data+'/filtered_data.pkl', "rb") as file:
    data = pickle.load(file)

with open(path_data+'/filtered_data_numeric.pkl', "rb") as file:
    data_numeric = pickle.load(file)

with open(path_modelo+'/model1.pkl', "rb") as file:
    model1 = pickle.load(file)



data_plots['perc_approved_sem1'] = data_plots['curricular_units_1st_sem_approved']/data_plots['curricular_units_1st_sem_enrolled']
data_plots['perc_approved_sem2'] = data_plots['curricular_units_2nd_sem_approved']/data_plots['curricular_units_2nd_sem_enrolled']
data_plots['Target'] = np.where(data_plots['target'] == 'Dropout', 1, 0)

# Calcula la matriz de correlación
correlation_matrix = data.corr()

filter['FAMI_ESTRATOVIVIENDA'].unique()
valid_values = ['Estrato 1', 'Estrato 2', 'Estrato 3',  'Estrato 4', 'Estrato 5', 'Estrato 6', 'Sin Estrato']

# Filter the DataFrame
filtered_data = data_numeric[data_numeric['FAMI_ESTRATOVIVIENDA'].isin(valid_values)]
category_order = {
    'FAMI_ESTRATOVIVIENDA': ['Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6', 'Sin Estrato']
}
# Crea el mapa de calor
fig1 = px.box(filtered_data, x='FAMI_ESTRATOVIVIENDA', y='PUNT_MATEMATICAS', 
             labels={'FAMI_ESTRATOVIVIENDA': 'Estrato Vivienda', 'PUNT_MATEMATICAS': 'Puntaje Matemáticas'},
             title="Distribución de Puntajes de Matemáticas por Estrato de Vivienda",
             category_orders=category_order)
#fig1.update_layout(autosize=False, width=800, height=600)
#fig1.show()

# Calcula los puntajes promedio por categoría
avg_scores = filtered_data.groupby('COLE_BILINGUE')['PUNT_GLOBAL'].mean().reset_index()

fig2 = px.bar(avg_scores, x='COLE_BILINGUE', y='PUNT_GLOBAL',
             labels={'COLE_BILINGUE': 'Colegio Bilingüe', 'PUNT_GLOBAL': 'Puntaje Global Promedio'},
             title="Puntaje Global Promedio en Colegios Bilingües vs No Bilingües")

# Asumiendo que 'data' es tu DataFrame
filtered_data['COLE_JORNADA'] = filtered_data['COLE_JORNADA'].replace('MAÃ‘ANA', 'MAÑANA')

filtered_data['COLE_JORNADA'].unique()
# Calcula los puntajes promedio para Matemáticas
avg_scores_math = filtered_data.groupby('COLE_JORNADA')['PUNT_MATEMATICAS'].mean().reset_index()
avg_scores_math['Materia'] = 'Matemáticas'

# Calcula los puntajes promedio para Lectura Crítica
avg_scores_reading = filtered_data.groupby('COLE_JORNADA')['PUNT_LECTURA_CRITICA'].mean().reset_index()
avg_scores_reading['Materia'] = 'Lectura Crítica'

filtered_data.columns
# Calcula los puntajes promedio para Lectura Crítica
avg_scores_science = filtered_data.groupby('COLE_JORNADA')['PUNT_C_NATURALES'].mean().reset_index()
avg_scores_science['Materia'] = 'Ciencias Naturales'

avg_scores_social = filtered_data.groupby('COLE_JORNADA')['PUNT_SOCIALES_CIUDADANAS'].mean().reset_index()
avg_scores_social['Materia'] = 'Ciencias Sociales'
avg_scores_ingles = filtered_data.groupby('COLE_JORNADA')['PUNT_INGLES'].mean().reset_index()
avg_scores_ingles['Materia'] = 'Inglés'

# Asegúrate de que ambas tablas tienen la misma estructura de columnas
avg_scores_reading = avg_scores_reading.rename(columns={'PUNT_LECTURA_CRITICA': 'Puntaje'})
avg_scores_math = avg_scores_math.rename(columns={'PUNT_MATEMATICAS': 'Puntaje'})
avg_scores_science = avg_scores_science.rename(columns={'PUNT_C_NATURALES': 'Puntaje'})
avg_scores_social = avg_scores_social.rename(columns={'PUNT_SOCIALES_CIUDADANAS': 'Puntaje'})
avg_scores_ingles = avg_scores_ingles.rename(columns={'PUNT_INGLES': 'Puntaje'})

# Combinamos los datos
combined_data = pd.concat([avg_scores_math, avg_scores_reading, avg_scores_science, avg_scores_social, avg_scores_ingles])

# Creamos el gráfico
fig3 = px.bar(combined_data, 
             x='COLE_JORNADA', 
             y='Puntaje', 
             color='Materia',
             barmode='group',
             labels={'COLE_JORNADA': 'Jornada', 'Puntaje': 'Puntaje Promedio'},
             title='Puntajes Promedio en las diferentes áreas del conocimiento por Jornada Escolar')

# Ajustamos el layout
#fig.update_layout(autosize=False, width=1000, height=600)

# Mostramos el gráfico
#fig.show()



# Unique values for dropdowns
unique_values = {
    "Age at enrollment": data["age_at_enrollment"].unique(),
    "Unemployment rate": data["unemployment_rate"].unique(),
    "Inflation rate": data["inflation_rate"].unique(),
    "Debtor": data["debtor"].unique(),
    "Scholarship holder": data["scholarship_holder"].unique(),
}

unique_values_2 = {
        "Scholarship holder": data["scholarship_holder"].unique(),
}

# Dictionary to map original values to more descriptive ones
readable_values = {
    "Age at enrollment": "Age Quartile",
    "Unemployment rate": {
        "Q1": "Low Unemployment",
        "Q2": "Below Average Unemployment",
        "Q3": "Above Average Unemployment",
        "Q4": "High Unemployment"
    },
    "Inflation rate": {
        "Q1": "Low Inflation",
        "Q2": "Below Average Inflation",
        "Q3": "Above Average Inflation",
        "Q4": "High Inflation"
    },
    "Debtor": {
        0: "No Debt",
        1: "Has Debt"
    },
    "Scholarship holder": {
        0: "No Scholarship",
        1: "Has Scholarship"
    }
}

dropdown_values = {}

for column, values in unique_values.items():
    if isinstance(readable_values[column], dict):
        dropdown_values[column] = [{'label': readable_values[column].get(value, value), 'value': value} for value in values]
    else:
        dropdown_values[column] = [{'label': value, 'value': value} for value in values]

# Adjust your app layout and callback accordingly...
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label='Prediction', children=[
            html.H1("Bayesian Network Prediction for Student Dropout"),
    
        # Description for the dashboard
            html.Div([
                html.P("This tool predicts the likelihood of a student dropping out based on various features."),
                html.P("Choose the relevant values from the dropdown menus below and click 'Predict'."),
                html.P("Descriptions:"),
                html.Ul([
                    html.Li("Age Quartile: Age range the student belongs to: Q1 (<19), Q2 (19-20), Q3 (20-25) and Q4 (25>)"),
                    html.Li("Unemployment Rate: Quartile of unemployment rate during admission."),
                    html.Li("Inflation Rate: Quartile of inflation rate during admission."),
                    html.Li("Debtor: Indicates if the student has debt for funding the university."),
                    html.Li("Scholarship Holder: Indicates if the student has a scholarship.")
                ])
            ], style={"border": "1px solid #ddd", "padding": "10px", "margin-bottom": "20px", "border-radius": "5px"}),
        
            # Dropdowns for evidence
            dbc.Row([dbc.Col([html.Label(column), dcc.Dropdown(id=column, options=dropdown_values[column], value=values[0])]) for column, values in unique_values.items()]),
            html.Br(),

            # Button to predict
            dbc.Button("Predict Dropout Model 1", id="predict-button", color="primary"),

            html.Br(), html.Br(),

            # Display results
            html.Div(id="prediction-result"),

            # Button to predict
            dbc.Button("Predict Dropout Model 2", id="predict-button-hill", color="primary"),

            html.Br(), html.Br(),

            # Display results
            html.Div(id="prediction-result-hill")
        ]),
        dbc.Tab(label='Visualizations', children=[
            html.H1("Data Insights"),
            dcc.Graph(id='age-target-plot', figure=fig1),
            dcc.Graph(id='units-approved-target-plot', figure=fig2),
            dcc.Graph(id='units-approved-debtor-plot', figure=fig3)
        ])
    ])

    
])

@app.callback(
    Output("prediction-result", "children"),
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State(column, "value") for column in unique_values]
)
def predict(n_clicks, age_enrollment, unemployment_rate, inflation_rate, debtor, scholarship_holder):
    if n_clicks:
        inference = VariableElimination(model1)
        prob = inference.query(
            variables=["actual_target"],
            evidence={
                "Age at enrollment": age_enrollment,
                "Unemployment rate": unemployment_rate,
                "Inflation rate": inflation_rate,
                "Debtor": debtor,
                "Scholarship holder": scholarship_holder
            })
        
        predicted_prob = prob.values[1]
        predicted_label = "Model 1 predicts the student will Dropout" if predicted_prob > 0.28697751471555144 else "Model predicts the student will not Dropout"
        
        return dbc.Alert([
            html.H4(f"Probability of Dropout: {predicted_prob:.4f}"),
            html.P(predicted_label, className="mb-0")
        ], color="success" if predicted_prob > 0.28697751471555144 else "warning")

    return ""


@app.callback(
    Output("prediction-result-hill", "children"),
    [Input("predict-button-hill", "n_clicks")],
    [dash.dependencies.State(column, "value") for column in unique_values_2]
)

def predict_hill(n_clicks, scholarship_holder):
    if n_clicks:
        inference = VariableElimination(model2)
        prob = inference.query(
            variables=["actual_target"],
            evidence={
                "Scholarship holder": scholarship_holder
            })
        
        predicted_prob = prob.values[1]
        predicted_label = "Model 2 predicts the student will Dropout" if predicted_prob >= 0.3797150041911148 else "Model predicts the student will not Dropout"
        
        return dbc.Alert([
            html.H4(f"Probability of Dropout: {predicted_prob:.4f}"),
            html.P(predicted_label, className="mb-0")
        ], color="success" if predicted_prob > 0.28697751471555144 else "warning")


    return ""

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True)
