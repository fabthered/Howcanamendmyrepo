test
# Données d'entrée par défaut
#test_size=0.35
#C=0.5
#penalty='l1'

###############################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div( # main frame
                [
                    html.Div( # header
                        [
                            html.H2(
                                "Supervised machine learning",
                                style={"margin-bottom": "0px",
                                       'textAlign': 'center',
                                       'color': '#3240a8'}
                            ),
                            html.H4(
                                "Methods comparison",
                                style={"margin-top": "0px",
                                       'textAlign': 'center',
                                       'color': '#7a83bf'}
                            ),
                        ],
                        className="row",
                        id="title",
                        ),
                                                                
                    html.Div( # 2 sub-frames
                        [
                            html.Div( # Left sub frame - input window
                                [
                                    dcc.Upload(id='upload-data', # file selection
                                        children=
                                            html.Div(['Drag and Drop or ',html.A('Select Files')]),
                                                style={
                                                    'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px'
                                                        },
                                                # Allow multiple files to be uploaded
                                                multiple=True
                                                ),
                                                                                  
                                    html.Div(id='output-data-upload'),
                                    
                                    html.Div(
                                        [
                                            html.Div([html.H6('Suggested methods')],
                                                id='suggested_methods',
                                                style={'display': 'block'}),
                                            
                                            
                                            dcc.Dropdown(id='model_selection',
                                                options=[
                                                    {'label': 'Logistic regression', 'value': 'LR'},
                                                    {'label': 'Decision tree', 'value': 'DT'},
                                                    {'label': 'SVM', 'value': 'SVM'},
                                                    {'label': 'Linear regression', 'value': 'LM'},
                                                    {'label': 'Polynomial regression', 'value': 'PR'},
                                                    {'label': 'KNN regression', 'value': 'KNN'}
                                                ],
                                                multi=True
                                                        ), 
        
                                            html.Div(
                                                [
                                                    
                                                    html.Div(
                                                        [
                                                            html.H6('Tuning the hyper parameters')
                                                         ],
                                                          id='hyperparameters',
                                                          className="pretty_container six columns"),
                                                        
                                                    html.Div(
                                                      [
                                                            dcc.RadioItems(
                                                                options=[
                                                                    {'label': 'Best fit', 'value': 'b'},
                                                                    {'label': 'User selected', 'value': 'u'}
                                                                ],
                                                                value='b',
                                                                labelStyle={'display': 'inline-block'},
                                                                style={
                                                                'width': '100%',
                                                                'height': '25px',
                                                                'margin': '12px'
                                                                    },
                                                                id="hyper_selection"),
                                                        ],
                                                        className="pretty_container six columns")
                                                ],
                                                className="row"
                                                ),
                                            
                                            
                                            html.Div(
                                                [
                                            
                                                html.Label('Logistic Regression'),
                                                          
                                                html.Div(
                                                    [
                                                        html.Div([html.Div(id='LR_C')],className="pretty_container five columns"),
                                                        
                                                        html.Div(
                                                            [
                                                                dcc.Slider(
                                                                id='LR_C_value',
                                                                min=0,
                                                                max=5,
                                                                step=0.1,
                                                                value=1,
                                                                    )      
                                                            ],
                                                            className="pretty_container seven columns",
                                                            style= {'display': 'block'})
                                                    ],
                                                    className='row'),
                                                                                                      
                                                html.Div(
                                                    [
                                                        html.Div([html.Div(id='LR_penalty')],className="pretty_container five columns"),
                                                        
                                                        html.Div(
                                                            [
                                                                dcc.Dropdown(
                                                                id='LR_penalty_value',
                                                                options=[{'label':'l1', 'value':'l1'},
                                                                         {'label':'l2', 'value':'l2'}],
                                                                value='l1'
                                                                )

                                                            ],
                                                            className="pretty_container seven columns")
                                                    ],
                                                    className='row'),
                                                
                                               html.Br(),
                                               
                                               html.Label('Decision Tree'),
                                                          
                                                html.Div(
                                                    [
                                                        html.Div([html.Div(id='DT_mln')],className="pretty_container five columns"),
                                                        
                                                        html.Div(
                                                            [
                                                                dcc.Slider(
                                                                id='DT_mln_value',
                                                                min=0,
                                                                max=100,
                                                                step=1,
                                                                value=5,
                                                                    )      
                                                            ],
                                                            className="pretty_container seven columns")
                                                    ],
                                                    className='row'),
                                                            
                                                html.Label('SVM'),
                                                          
                                                html.Div(
                                                    [
                                                        html.Div([html.Div(id='SVM_C')],className="pretty_container five columns"),
                                                        
                                                        html.Div(
                                                            [
                                                                dcc.Slider(
                                                                id='SVM_C_value',
                                                                min=0.1,
                                                                max=1000,
                                                                step=0.1,
                                                                value=1,
                                                                    )      
                                                            ],
                                                            className="pretty_container seven columns")
                                                    ],
                                                    className='row'),
                                                            
                                                html.Div(
                                                    [
                                                        html.Div([html.Div(id='SVM_gamma')],className="pretty_container five columns"),
                                                        
                                                        html.Div(
                                                            [
                                                                dcc.Slider(
                                                                id='SVM_gamma_value',
                                                                min=0.0001,
                                                                max=1,
                                                                step=0.0001,
                                                                value=0.1,
                                                                    )      
                                                            ],
                                                            className="pretty_container seven columns")
                                                    ],
                                                    className='row'),     
            
                                                html.Div(
                                                    [
                                                        html.Div([html.Div(id='SVM_ker')],className="pretty_container five columns"),

                                                        html.Div(
                                                            [
                                                                dcc.Dropdown(
                                                                id='SVM_kernel',
                                                                options=[{'label':'linear', 'value':'linear'},
                                                                         {'label':'rbf', 'value':'rbf'}],
                                                                value='linear'
                                                                )

                                                            ],
                                                            className="pretty_container seven columns")
                                                    ],
                                                    className='row'),
                                                
                                                html.Br(),
                                                
                                                html.Label('KNN Regression'),
                                                          
                                                html.Div(
                                                    [
                                                        html.Div([html.Div(id='KNN_n_value')],className="pretty_container five columns"),
                                                        
                                                        html.Div(
                                                            [
                                                                dcc.Slider(
                                                                id='KNN_n',
                                                                min=0,
                                                                max=19,
                                                                step=1,
                                                                value=5,
                                                                    )      
                                                            ],
                                                            className="pretty_container seven columns")
                                                    ],
                                                    className='row')
                                                ],
                                                id='user_selection',
                                                style= {'display': 'block'}
                                                ),
                                                            
                                                html.Button(id='reset2', n_clicks=0, children='Reset', style={'display': 'block'}),

                                                                                
                                            # Hidden div inside the app that stores the intermediate value df
                                            html.Div(id='intermediate-value', style={'display': 'none'})
                                        ],
                                        id="method_selection",
                                        style={'display': 'none'}
                                        )
                                     ],   
                                    className="one-third column"),
                           
                        html.Div( # Right sub frame - input window
                            [
                                html.H4(
                                    "Results and evaluation metrics",
                                    style={"margin-top": "0px"}
                                    ),
                                
                                #html.Button(id='submit-algos', n_clicks=0, children='Compare the selected methods', style={'display': 'block'}),
                                
                                dcc.Loading(
                                            id="loading-LR",
                                            type="default",
                                            children=html.Div(id="loading-output-LR", style={'display': 'none'})
                                            ),
                                dcc.Loading(
                                            id="loading-DT",
                                            type="default",
                                            children=html.Div(id="loading-output-DT", style={'display': 'none'})
                                            ), 
                                dcc.Loading(
                                            id="loading-SVM",
                                            type="default",
                                            children=html.Div(id="loading-output-SVM", style={'display': 'none'})
                                            ),
                                dcc.Loading(
                                            id="loading-LM",
                                            type="default",
                                            children=html.Div(id="loading-output-LM", style={'display': 'none'})
                                            ),
#                                dcc.Loading(
#                                            id="loading-PR",
#                                            type="default",
#                                            children=html.Div(id="loading-output-PR", style={'display': 'none'})
#                                            ),
#                                dcc.Loading(
#                                            id="loading-KNN",
#                                            type="default",
#                                            children=html.Div(id="loading-output-KNN", style={'display': 'none'})
#                                            ),
                                
                                html.Br(),
                                
                                dcc.Tabs(
                                    [        
                                        dcc.Tab(label='Logistic Regression',
                                                children=[

                                                html.Div(
                                                    [
                                                        html.H6('Elapsed time'),
                                                        html.Div(id='LR_elapsed_time')
                                                    ],
                                                    className="pretty_container three columns"),
                                                        
                                                html.Div(
                                                    [
                                                        html.H6('Evaluation metrics'),
                                                        html.Div(id='LR_evaluation_metrics'),
                                                        html.Div(id='LR_accuracy_score'),
                                                        html.Div(id='LR_precision_score'),
                                                        html.Div(id='LR_recall_score'),
                                                        html.Div(id='LR_f1_score_CV')
                                                     ],
                                                    className="pretty_container three columns"),
                                                    
                                                html.Div(
                                                    [
                                                        html.H6('Hyper parameters'),
                                                        html.Div(id='LR_params_C'),
                                                        html.Div(id='LR_params_penalty')
                                                     ],
                                                    className="pretty_container three columns"),
                                                                                                    
                                                html.Div(
                                                    [
                                                        html.H6('ROC curve'),
                                                        dcc.Graph(id='LR_roc_fig')
                                                     ],
                                                    className="pretty_container three columns")
                                            ]),
                                                
                                        dcc.Tab(label='Decision Tree',
                                                children=[
                                                    html.Div(
                                                    [
                                                        html.H6('Elapsed time'),
                                                        html.Div(id='DT_elapsed_time')
                                                    ],
                                                    className="pretty_container three columns"),
                                                        
                                                html.Div(
                                                    [
                                                        html.H6('Evaluation metrics'),
                                                        html.Div(id='DT_evaluation_metrics'),
                                                        html.Div(id='DT_accuracy_score'),
                                                        html.Div(id='DT_precision_score'),
                                                        html.Div(id='DT_recall_score'),
                                                        html.Div(id='DT_f1_score_CV')
                                                     ],
                                                    className="pretty_container three columns"),
                                                    
                                                html.Div(
                                                    [
                                                        html.H6('Hyper parameters'),
                                                        html.Div(id='DT_param_mln')
                                                     ],
                                                    className="pretty_container three columns")
                                            ]),
                                              
                                        dcc.Tab(label='SVM',
                                                children=[
                                                    html.Div(
                                                    [
                                                        html.H6('Elapsed time'),
                                                        html.Div(id='SVM_elapsed_time')
                                                    ],
                                                    className="pretty_container three columns"),
                                                        
                                                html.Div(
                                                    [
                                                        html.H6('Evaluation metrics'),
                                                        html.Div(id='SVM_evaluation_metrics'),
                                                        html.Div(id='SVM_accuracy_score'),
                                                        html.Div(id='SVM_precision_score'),
                                                        html.Div(id='SVM_recall_score'),
                                                        html.Div(id='SVM_f1_score_CV')
                                                     ],
                                                    className="pretty_container three columns"),
                                                    
                                                html.Div(
                                                    [
                                                        html.H6('Hyper parameters'),
                                                        html.Div(id='SVM_params_C'),
                                                        html.Div(id='SVM_params_gamma'),
                                                        html.Div(id='SVM_params_kernel')
                                                     ],
                                                    className="pretty_container three columns")
                                            ]),  
                                                
                                        dcc.Tab(label='Linear regression',
                                                children=[
                                                    html.Div(
                                                    [
                                                        html.H6('Elapsed time'),
                                                        html.Div(id='LM_elapsed_time')
                                                    ],
                                                    className="pretty_container three columns"),
                                                        
                                                html.Div(
                                                    [
                                                        html.H6('Evaluation metrics'),
                                                        html.Div(id='LM_evaluation_metrics'),
                                                        html.Div(id='LM_rmse'),
                                                        html.Div(id='LM_r2'),
                                                        html.Div(id='LM_f1_score_CV')
                                                     ],
                                                    className="pretty_container three columns"),
                                                    
                                                html.Div(
                                                    [
                                                        html.H6('Hyper parameters'),
                                                        html.Div(id='LM_params')
                                                     ],
                                                    className="pretty_container three columns"),
                                                
                                                html.Div(
                                                    [
                                                        html.H6('Correlation plot'),
                                                        dcc.Graph(id='LM_corr_plot')
                                                     ],
                                                    className="pretty_container three columns"),
                                                
                                                html.Div(
                                                    [
                                                        html.H6('Prediction plot'),
                                                        dcc.Graph(id='LM_pred_plot')
                                                     ],
                                                    className="pretty_container three columns"),
                                                
                                                html.Div(
                                                    [
                                                        html.H6('Residus plot'),
                                                        dcc.Graph(id='LM_residus_plot')
                                                     ],
                                                    className="pretty_container three columns")
                                            ]), 
                                                
                                        dcc.Tab(label='Polynomial regression',
                                                children=[
                                                    html.Div(
                                                    [
                                                        html.H6('Elapsed time'),
                                                        html.Div(id='PR_elapsed_time')
                                                    ],
                                                    className="pretty_container three columns"),
                                                        
                                                html.Div(
                                                    [
                                                        html.H6('Evaluation metrics'),
                                                        html.Div(id='PR_evaluation_metrics'),
                                                        html.Div(id='PR_accuracy_score'),
                                                        html.Div(id='PR_precision_score'),
                                                        html.Div(id='PR_recall_score'),
                                                        html.Div(id='PR_f1_score_CV')
                                                     ],
                                                    className="pretty_container three columns"),
                                                    
                                                html.Div(
                                                    [
                                                        html.H6('Hyper parameters'),
                                                        html.Div(id='PR_params')
                                                     ],
                                                    className="pretty_container three columns")
                                            ]),
                                                
                                        dcc.Tab(label='KNN regression',
                                                children=[
                                                    html.Div(
                                                    [
                                                        html.H6('Elapsed time'),
                                                        html.Div(id='KNN_elapsed_time')
                                                    ],
                                                    className="pretty_container three columns"),
                                                        
                                                html.Div(
                                                    [
                                                        html.H6('Evaluation metrics'),
                                                        html.Div(id='KNN_evaluation_metrics'),
                                                        html.Div(id='KNN_accuracy_score'),
                                                        html.Div(id='KNN_precision_score'),
                                                        html.Div(id='KNN_recall_score'),
                                                        html.Div(id='KNN_f1_score_CV')
                                                     ],
                                                    className="pretty_container three columns"),
                                                    
                                                html.Div(
                                                    [
                                                        html.H6('Hyper parameters'),
                                                        html.Div(id='KNN_param_n')
                                                     ],
                                                    className="pretty_container three columns")
                                            ]),                                                
                                                
                                        ],
                                        className="row")

                        ],
                            className="pretty_container seven columns"
                            )
                    ])
                ])                                    
                                  
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            global df
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
            html.H5(filename),
            
            # RadioItems de la variable cible
            html.H6('Select the target variable (last column by default)',id="l2"),
            dcc.Dropdown(id='vc-radioitems',
                options=[{'label': v, 'value': v} for v in df.columns],
                value=df.columns.values[-1],
                multi=False),
                         
            html.Br(),
            # Checklist des Variables prédictives
            html.H6('Select the feature variables',id="l1"),
            dcc.Dropdown(id="vp-checklist",
                    options=[{'label': v, 'value': v} for v in df.columns],
                    value=df.columns.values[:-1],
                    multi=True),                          
                       
            html.Br(),                       
            
        html.Button(id='proceed', n_clicks=0, children='Proceed to next step', style={'display': 'block'}),
        html.Br(),
        html.Button(id='reset1', n_clicks=0, children='Reset', style={'display': 'block'}),

        #html.Div(id='output-state')
        ],id="div_var",style={'display': 'block'}) 


#callback on the upload
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


#On feature variables selection, we remove it from the target variables list
@app.callback(
    [Output('vc-radioitems', 'options'),
    Output('vc-radioitems', 'value')],
    [Input('vp-checklist', 'value')],
    [State('vp-checklist', 'options'),
    State('vc-radioitems', 'value'),
    State('vc-radioitems', 'options')])
def set_target_options_value(selected_var,options_vp,v_vc,options_vc):
    options2=[]
    var_p=[]
    valtarget=v_vc
    # updating the target variables depending on the predictors checked
    if (selected_var!=None):
        for i in options_vp:
            if (i["value"] not in selected_var):
                options2.append(i)
                var_p.append(i["value"])
        #update predictors
        global X
        X=df.loc[:,var_p]
        #initializing target variable if it was chosen as predictor instead
        if (v_vc in selected_var):
            valtarget=options2[0]["value"]  
    else:
        options2=options_vp
        #No predictors
        X=pd.DataFrame()
    #update target
    global y
    y=df[valtarget]
    return options2,valtarget

#Determines whether the target is categorical or continous
def get_target_type(target):
    import numpy as np
    if(y.dtype == np.float64 or y.dtype == np.int64):
    #if(df[target].dtype == np.float64 or df[target].dtype == np.int64):
        return "num"
    else:
        return "cat"

#We update the models depending on target (class or regr)
@app.callback(
    Output('model_selection','options'),
    [Input('vc-radioitems', 'value')],prevent_initial_call=True)
def update_model_selection(selected_targ):
    global y
    y=df[selected_targ]
    options_class=[
    {'label': 'Logistic regression', 'value': 'LR'},
    {'label': 'Decision tree', 'value': 'DT'},
    {'label': 'SVM', 'value': 'SVM'}
    ]
    options_reg=[
    {'label': 'Linear regression', 'value': 'LM'},
    {'label': 'Polynomial regression', 'value': 'PR'},
    {'label': 'KNN regression', 'value': 'KNN'}
    ]
    model_options=dict()
    if (get_target_type(y)=="num"):
        model_options=options_reg
    else:
        model_options=options_class
    
    return model_options

# Run the calculus if LR is selected
@app.callback([Output(component_id='LR_elapsed_time', component_property='children'),
               Output(component_id='LR_accuracy_score', component_property='children'),
               Output(component_id='LR_precision_score', component_property='children'),
               Output(component_id='LR_recall_score', component_property='children'),
               Output(component_id='LR_f1_score_CV', component_property='children'),
               Output(component_id='LR_params_C', component_property='children'),
               Output(component_id='LR_params_penalty', component_property='children'),
               Output(component_id='LR_roc_fig', component_property='figure'),               
               Output("loading-output-LR", "children")],
               [Input('model_selection', 'value'),
                Input(component_id='hyper_selection', component_property='value'),
                Input(component_id='LR_C_value', component_property='value'),
                Input(component_id='LR_penalty_value', component_property='value')])
def running_LR(value1, value2, value3, value4):  
    #turn the target variable into binary
    global y
    #y = y.replace({'M':1, 'B':0})
    class_values=y.unique()
    y = y.replace({class_values[0] :1, class_values[1]:0})
    if ('LR' in value1): 
        from learn_functions import LR_pred
        if value2 == 'b':
            LR = LR_pred(X,y)
        if value2 == 'u':
            LR = LR_pred(X,y, C=value3, penalty=value4)
        container01="Elapsed time : {:.2f}".format(LR['elapsed_time'])+"s"
        container02="Accuracy score : {:.3f}".format(LR['accuracy_score'])
        container03="Precision score : {:.3f}".format(LR['precision_score'])
        container04="Recall score : {:.3f}".format(LR['recall_score'])
        container05="F1 score CV : {:.3f}".format(LR['f1_score_CV'])
        container06="C : {}".format(LR['clf.best_params_C'])
        container07="Penalty : {}".format(LR['clf.best_params_penalty'])
        container08=LR['roc_fig']
        container09=0     
        return container01, container02, container03, container04, container05, container06, container07, container08, container09
    else:    
        return '','','','','','','','',0

# Run the calculus if DT is selected
@app.callback([Output(component_id='DT_elapsed_time', component_property='children'),
               Output(component_id='DT_accuracy_score', component_property='children'),
               Output(component_id='DT_precision_score', component_property='children'),
               Output(component_id='DT_recall_score', component_property='children'),
               Output(component_id='DT_f1_score_CV', component_property='children'),
               Output(component_id='DT_param_mln', component_property='children'),              
               Output("loading-output-DT", "children")],
               [Input('model_selection', 'value'),
                Input(component_id='hyper_selection', component_property='value'),
                Input(component_id='DT_mln_value', component_property='value')])
def running_DT(value1, value2, value3):  
    if ('DT' in value1):     
        from learn_functions import DecisionTree
        if value2 == 'b':
            DT = DecisionTree(X,y)
        if value2 == 'u':
            DT = DecisionTree(X,y, max_leaf_nodes=value3)
        container01="Elapsed time : {:.2f}".format(DT['elapsed_time'])+"s"
        container02="Accuracy score : {:.3f}".format(DT['accuracy_score'])
        container03="Precision score : {:.3f}".format(DT['precision_score'])
        container04="Recall score : {:.3f}".format(DT['recall_score'])
        container05="F1 score CV : {:.3f}".format(DT['f1_score_CV'])
        container06="Max leaf nodes : {}".format(DT['max_leaf_nodes'])
        container07=0     
        return container01, container02, container03, container04, container05, container06, container07
    else:    
        return '','','','','','',0

# Run the calculus if SVM is selected
@app.callback([Output(component_id='SVM_elapsed_time', component_property='children'),
               Output(component_id='SVM_accuracy_score', component_property='children'),
               Output(component_id='SVM_precision_score', component_property='children'),
               Output(component_id='SVM_recall_score', component_property='children'),
               Output(component_id='SVM_f1_score_CV', component_property='children'),
               Output(component_id='SVM_params_C', component_property='children'),
               Output(component_id='SVM_params_gamma', component_property='children'),
               Output(component_id='SVM_params_kernel', component_property='children'),
               Output(component_id='loading-output-SVM',component_property='children')],
               [Input(component_id='model_selection',component_property='value'),
                Input(component_id='hyper_selection', component_property='value'),
                Input(component_id='SVM_C_value', component_property='value'),
                Input(component_id='SVM_gamma_value', component_property='value'),
                Input(component_id='SVM_kernel', component_property='value')])
def running_SVM(value1, value2, value3, value4, value5):  
    if ('DT' in value1):     
        from learn_functions import svm
        if value2 == 'b':
            SVM = svm(X,y)
        if value2 == 'u':
            SVM = svm(X,y,C=value3, gamma=value4, kernel=value5)
        container01="Elapsed time : {:.2f}".format(SVM['elapsed_time'])+"s"
        container02="Accuracy score : {:.3f}".format(SVM['accuracy_score'])
        container03="Precision score : {:.3f}".format(SVM['precision_score'])
        container04="Recall score : {:.3f}".format(SVM['recall_score'])
        container05="F1 score CV : {:.3f}".format(SVM['f1_score_CV'])
        container06="C : {}".format(SVM['clf.best_params_C'])
        container07="Gamma : {}".format(SVM['clf.best_params_gamma'])
        container08="Kernel : {}".format(SVM['clf.best_params_kernel'])
        container09=0     
        return container01, container02, container03, container04, container05, container06, container07, container08, container09
    else:    
        return '','','','','','','','',0
    
# Run the calculus if LinearRegression is selected
@app.callback([Output(component_id='LM_elapsed_time', component_property='children'),
               Output(component_id='LM_rmse', component_property='children'),
               Output(component_id='LM_r2', component_property='children'),
               Output(component_id='LM_f1_score_CV', component_property='children'),
               Output(component_id='LM_corr_plot', component_property='figure'),
               Output(component_id='LM_pred_plot', component_property='figure'), 
               Output(component_id='LM_residus_plot', component_property='figure'),                
               Output("loading-output-LM", "children")],
               [Input('model_selection', 'value'),
                Input(component_id='hyper_selection', component_property='value')])
def running_LM(value1, value2):  
    if ('LM' in value1):     
        from learn_functions import LinearRegression
        if value2 == 'b':
            LM = LinearRegression(X,y)
        if value2 == 'u':
            LM = LinearRegression(X,y)
        container01="Elapsed time : {:.2f}".format(LM['elapsed_time'])+"s"
        container02="RMSE : {:.3f}".format(LM['rmse'])
        container03="R2 : {:.3f}".format(LM['r2'])
        container04="F1 score CV : {:.3f}".format(LM['mean_ols_cv_mse'])
        container05=LM['plot_correlation']
        container06=LM['plot_prediction']
        container07=LM['plot_residus']
        container08=0     
        return container01, container02, container03, container04, container05, container06, container07, container08
    else:    
        return '','','','','','','',0

@app.callback(
    [Output('LR_C', 'children'),
    Output('LR_penalty', 'children'),
    Output('DT_mln', 'children'),
    Output('SVM_C', 'children'),
    Output('SVM_gamma', 'children'),
    Output('SVM_ker', 'children'),
    Output('KNN_n_value', 'children')],
    [Input('LR_C_value', 'value'),
     Input('LR_penalty_value', 'value'),
     Input('DT_mln_value', 'value'),
     Input('SVM_kernel', 'value'),
     Input('SVM_C_value', 'value'),
     Input('SVM_gamma_value', 'value'),
     Input('KNN_n', 'value')])
def update_LR_C(C_value, penalty_value, mln_value, C_value2, gamma_value, kernel_value, n_value):
    return 'C = {}'.format(C_value), 'Penalty : "{}"'.format(penalty_value), 'Max leaf nodes = {}'.format(mln_value), 'C = {}'.format(C_value2), 'Gamma = {}'.format(gamma_value), 'Kernel : {}'.format(kernel_value), 'KNN neighbors = {}'.format(n_value)


@app.callback(Output(component_id='user_selection', component_property='style'),
    [Input(component_id='hyper_selection', component_property='value')])
def update_x(user_selection):
    if user_selection == 'u':
        return {'display': 'block'}
    if user_selection == 'b':
        return {'display': 'none'}

@app.callback([Output(component_id='method_selection', component_property='style'),
               Output(component_id='div_var', component_property='style')],
                [Input('proceed', 'n_clicks')])
def update_method_selection(n_clicks):
    if n_clicks > 0:
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

if __name__ == '__main__':
    app.run_server(debug=True)
  
    
