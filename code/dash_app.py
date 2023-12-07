

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
from pgmpy.inference import VariableElimination

data= pd.read_pickle('/opt/app/filtered_data.pkl')
data_numeric= pd.read_pickle('/opt/app/filtered_data_numeric.pkl')

model1 = pd.read_pickle('/opt/app/model1.pkl')


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

# Calcula los puntajes promedio para Matemáticas
avg_scores_math = filtered_data.groupby('COLE_JORNADA')['PUNT_MATEMATICAS'].mean().reset_index()
avg_scores_math['Materia'] = 'Matemáticas'

# Calcula los puntajes promedio para Lectura Crítica
avg_scores_reading = filtered_data.groupby('COLE_JORNADA')['PUNT_LECTURA_CRITICA'].mean().reset_index()
avg_scores_reading['Materia'] = 'Lectura Crítica'

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


data['COLE_JORNADA'] = data['COLE_JORNADA'].replace('MAÃ‘ANA', 'MAÑANA')

# Unique values for dropdowns
unique_values = {
    "COLE_JORNADA": data["COLE_JORNADA"].unique(),
    "COLE_CALENDARIO": data['COLE_CALENDARIO'].unique(),
    "COLE_BILINGUE": data['COLE_BILINGUE'].unique(),
}

# Dictionary to map original values to more descriptive ones
readable_values = {


}

#dropdown_values = {}

#for column, values in unique_values.items():
#    if isinstance(readable_values[column], dict):
#        dropdown_values[column] = [{'label': readable_values[column].get(value, value), 'value': value} for value in values]
#    else:
#        dropdown_values[column] = [{'label': value, 'value': value} for value in values]


dropdown_values = {}

dropdown_values = {}
for column, values in unique_values.items():
    dropdown_values[column] = [{'label': value, 'value': value} for value in values]


# Adjust your app layout and callback accordingly...
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server 

app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label='Predicción', children=[
            html.H1("¡Elija el mejor colegio para sus hijos!"),
    
        # Description for the dashboard
            html.Div([
                html.P("Esta herramienta permite predecir la probabilidad de que un estudiante se encuentre por encima del percentil 50 en el ICFES."),
                html.P("Seleccione los valores correspondientes de acuerdo a las características del colegio"),
                html.P("Descripciones"),
                html.Ul([
                    html.Li("Jornada Colegio: Tipo de Jornada del Colegio"),
                    html.Li("Calendario Colegio: Tipo de calendario del Colegio"),
                    html.Li("Colegio Bilingue: S si el colegio es Bilingue, N si no lo es")

                ])
            ], style={"border": "1px solid #ddd", "padding": "10px", "margin-bottom": "20px", "border-radius": "5px"}),
        
            # Dropdowns for evidence
            dbc.Row([dbc.Col([html.Label(column), dcc.Dropdown(id=column, options=dropdown_values[column], value=values[0])]) for column, values in unique_values.items()]),
            html.Br(),

            # Button to predict
            dbc.Button("Predecir Resultado", id="predict-button", color="primary"),

            html.Br(), html.Br(),

            # Display results
            html.Div(id="prediction-result"),
        ]),
        dbc.Tab(label='Visualizaciones', children=[
            html.H1("Algunos datos interesantes"),
            html.H2("¡Aquí podrá ver algunas visualizaciones que le permiten entender cuales son las claves del éxito académico de su hijo!"),

            dcc.Graph(id='fig1', figure=fig1),
            dcc.Graph(id='fig2', figure=fig2),
            dcc.Graph(id='fig3', figure=fig3)
        ])
    ])

    
])

@app.callback(
    Output("prediction-result", "children"),
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State(column, "value") for column in unique_values]
)

def predict(n_clicks, cole_jornada, cole_calendario, cole_bilingue):
    if n_clicks:
        cole_jornada = cole_jornada.replace('MAÑANA', 'MAÃ‘ANA')

        inference = VariableElimination(model1)
        prob = inference.query(
            variables=["PUNT_GLOBAL"],
            evidence={
                "COLE_JORNADA": cole_jornada,
                "COLE_CALENDARIO": cole_calendario,
                "COLE_BILINGUE": cole_bilingue
            })
        
        predicted_prob = prob.values[1]
        predicted_label = "El modelo predice que el estudiante obtendrá un resultado por encima del percentil 50" if predicted_prob > 0.504 else "El modelo predice que el estudiante obtendrá un resultado por debajo del percentil 50"
        
        return dbc.Alert([
            html.H4(f"Probabilidad de obtener un resultado por encima del percentil 50: {predicted_prob:.4f}"),
            html.P(predicted_label, className="mb-0")
        ], color="success" if predicted_prob > 0.504 else "warning")

    return ""

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True)
