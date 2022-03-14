# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from pydoc import classname
from turtle import title
from xmlrpc.server import DocCGIXMLRPCRequestHandler
from dash import Dash, html, dcc, dash_table, Output, Input
import requests
import json
import plotly.graph_objects as go
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np



# My custom css sheet
external_script = ["./files/mycss.css", {"src": "./files/mycss.css"}]
app = Dash(
    __name__,
    external_scripts=external_script,
)   
server = app.server


explainer =  pickle.load(open('./files/shap_explainer','rb'))
reducer =  pickle.load(open('./files/reducer','rb'))


# Dataframe
df = pd.read_csv('./files/application_test.csv', 
                usecols=['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','NAME_FAMILY_STATUS'])

df['Ratio_credit_revenu'] = df['AMT_INCOME_TOTAL']*100/(12*df['AMT_ANNUITY'])


app.layout = html.Div(id='my_app',children=[

    #Titre de la page
    html.Div(id='header',children=[
        html.H1(children='Prêt à dépenser'),
        html.H4(children='Le crédit pour tous! Simple, Rapide et Transparent'),
    ]),
    
    #Dasboard Body
    html.Div(children=[

        #Requetage
        html.Div(id='request',children=[
            html.Span(id='req',children=[
                html.P(children='Identifiant Client : '),
                html.P(children=dcc.Input(id='id_client',placeholder='ex 10025')),
            ]),
            
            html.Span(id='label'),
            html.Span(id='score'),
        ]),

        #Results requetage
        html.Div(id ='shap_plot', children=[
    
            
        ]),
    

        ###Graphiques
        html.Div(children=[
            
            html.H2(children='Insights' ),
            
            html.Div(id='graphique',children=[
                html.Div(className='panel',children=[

                    html.H4("Distribution des montants de crédis demandés"),
                    html.P("Déclinaison :"),

                    dcc.RadioItems(
                        id='y-axis', 
                        options={
                            'CODE_GENDER': 'sexe',
                            'NAME_FAMILY_STATUS':'statut matrimonial',
                            'NAME_CONTRACT_TYPE':'type de contrat',
                            'FLAG_OWN_CAR':'voiture'
                        },
                        value='CODE_GENDER', 
                        inline=True
                    ),

                    dcc.Dropdown(
                        id='x-axis', 
                        options={
                            'AMT_INCOME_TOTAL': 'Niveau de revenu',
                            'AMT_CREDIT':'Montant du crédit',
                            'Ratio_credit_revenu':'Ratio Crédit/Revenu',
                        },
                        value='AMT_INCOME_TOTAL', 
                    ),
                ]),

                html.Div(className='plot', children=[

                    dcc.Graph(id='profil_graphic')

                ]),

            ]),

        ])

    ])
])

# Profil graphic
@app.callback(
    Output(component_id='profil_graphic', component_property='figure'),
    Input(component_id='x-axis', component_property='value'),
    Input(component_id='y-axis', component_property='value')
)
def profil_graphic(xvar, yvar):
    '''
    for plotting amount of credit according a profil (gender, matrimonial statut, type of contrat)
    '''
    fig = px.box(df, y=yvar, x=xvar)
    fig.update_xaxes(title=xvar)
    fig.update_yaxes(title=yvar)
    return fig


# Request to API prediction
@app.callback(
    Output(component_id='label', component_property='children'),
    Output(component_id='score', component_property='children'),
    Output(component_id='shap_plot', component_property='children'),
    Input(component_id='id_client', component_property='value')
)
def prediction(id):
    '''
    This function run requests to model api and return label, score and shap plot
    '''
    if (id!=None) and (id!=''):
       # r = requests.post('URL_OF_API_MODEL/model?id='+id) 
       # r = requests.post('http://127.0.0.1:8000/model?id='+id)
       r = requests.post('https://crp-model-api.herokuapp.com/model?id='+id)
       pred=r.json()
       if pred.get('message')==None:
           ## Adding a meaning label
           if pred["label"] == 0:
               label='Solvable'
           else:
               label='Non solvable'
           data = pd.DataFrame.from_dict(json.loads(pred['d']))
           force_plot = shap.force_plot(explainer(data[reducer]), matplotlib=False)
           shap_html = f"<head>{shap.getjs()}</head><body style='background-color:white;'>{force_plot.html()}</body>"
           fig_shap = html.Iframe(srcDoc=shap_html, style={"width": "100%", "height": "175px", "padding":"5px 0px 5px 0px",
                       "border": 0})

        
         #       fig_shap=dcc.Graph(
          #          figure= shap.force_plot(pred['d'][0], feature_names=reducer, matplotlib=True, show=False)
           #         )
       else:
            label = pred["message"]
            pred["score"] = '-'
            fig_shap = html.Span(children='')
        
       return label, f"{pred['score']} %", fig_shap

    else:
        return f'',f'', f''

if __name__ == '__main__':
    app.run_server(debug=True)
