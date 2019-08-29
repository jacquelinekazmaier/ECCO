'''Contains functions for visualising and analysing model results by means of the Dashboard interface'''

import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import random, threading, webbrowser
import json
import base64
import pydotplus
import matplotlib as mpl
import nltk
import collections
import pyLDAvis

mpl.use('QT5Agg')

from imageio import imwrite as imsave
from dash.dependencies import Output, Input, State
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from io import StringIO
from wordcloud import WordCloud
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from pyLDAvis import gensim as gensimvis

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

##################################################
# Style
##################################################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colours = dict(background='#DBDFDC', text='#707070', sentiments=['#82DFAA', '#DE3B3B', '#F9E79F'],
               sentiment_dict={'positive': '#82DFAA', 'negative': '#DE3B3B', 'neutral': '#F9E79F'})


class Dashboard:
    def __init__(self, final_predictions, reviews, customer_data=None):
        self.final_predictions = final_predictions
        self.clients = customer_data
        self.reviews = reviews
        self.unified = pd.merge(self.final_predictions, self.clients.df, on='Client_num', how='inner')
        # note: above line removes entries without corresponding entries in supplementary data (clients)
        self.unified.to_csv('./data/unified.csv')

        self.qualitative_profiles = self.clients.qualitative_profiles
        self.qualitative_profiles.extend(reviews.qualitative_profiles)
        self.quantitative_profiles = self.clients.quantitative_profiles
        self.quantitative_profiles.extend(reviews.quantitative_profiles)

        self.date = reviews.date
        self.locations = reviews.locations

        if reviews.loc != None:
            self.locations.extend([reviews.loc])

        # remove unselected columns
        drop_list = set(self.unified.iloc[:, 3:].columns) - set(self.qualitative_profiles) - set(
            self.quantitative_profiles)
        try:
            drop_list = drop_list - set([self.date]) - set(self.locations)
        except:
            try:
                drop_list = drop_list - set([self.date])
            except:
                try:
                    drop_list = drop_list - set(self.locations)
                except:
                    pass
        self.unified.drop(drop_list, axis=1, inplace=True)

        df = self.unified
        self.sel_tokens = []
        for message in df['Review'].str.lower():
            tokenized = [i for i in nltk.word_tokenize(message) if i not in nltk.corpus.stopwords.words('english')]
            self.sel_tokens.append(tokenized)

    ##################################################
    # Create Graphs
    ##################################################
    def count_instances(self, list_of_labels):
        num_pos = sum(list_of_labels == 'positive')
        num_neg = sum(list_of_labels == 'negative')
        num_neut = sum(list_of_labels == 'neutral')
        return [num_pos, num_neg, num_neut]

    def create_wordcloud(self, text, path, stopwords):
        wordcloud = WordCloud(mode="RGBA", width=300, height=300, max_words=300,
                              background_color=None, stopwords=stopwords, scale=2).generate(text)
        imsave(path, wordcloud)
        encoded_cloud = base64.b64encode(open(path, 'rb').read())
        return encoded_cloud

    def make_piechart(self):
        labels = ['positive', 'negative', 'neutral']
        values = self.count_instances(self.unified['Sentiment'])

        piechart = dcc.Graph(id='piechart',
                             figure=go.Figure(data=[go.Pie(labels=labels, values=values,
                                                           hoverinfo='label+value', textinfo='percent',
                                                           marker=dict(colors=colours['sentiments']), opacity=0.8)],
                                              layout=go.Layout(title='Overall sentiment',
                                                                #plot_bgcolor = colours['background'],
                                                                #paper_bgcolor = colours['background'],
                                                               # font = 'color': colours['text']
                                                               )
                                              )
                             )
        return piechart

    def make_parallel_cat_diagram(self):
        # Create dimensions
        dims = []
        for dimension in self.qualitative_profiles:
            dims.append(go.parcats.Dimension(
                values=self.unified[dimension],
                label=dimension
            ))
        dims.append(go.parcats.Dimension(
            values=self.unified['Sentiment'],
            label='sentiment'
        )
        )
        # Create parcats trace
        new = []
        for c in self.unified['Sentiment']:
            if c == 'negative':
                new.append(0)
            elif c == 'neutral':
                new.append(0.5)
            else:
                new.append(1)
        colourscale = [[0, 'indianred'], [0.5, 'palegoldenrod'], [1, 'mediumseagreen']];

        data = [
            go.Parcats(
                dimensions=dims,
                line={'color': new,
                      'colorscale': colourscale},
                hoveron='color',
                hoverinfo='count+probability',
                labelfont={'size': 18, 'family': 'Times'},
                tickfont={'size': 16, 'family': 'Times'},
                arrangement='freeform'
            )
        ]

        parcat = dcc.Graph(id='parcat',
                           figure=go.Figure(data=data,
                                            layout=go.Layout(title='Customer profiles'
                                                             # plot_bgcolor = colours['background'],
                                                             # paper_bgcolor = colours['background'],
                                                             # font = 'color': colours['text']
                                                             )
                                            )
                           )
        return parcat

    def make_barchart(self):
        options = [{'label': 'No grouping', 'value': 'None'}]
        options.extend([{
            'label': i,
            'value': i
        } for i in self.qualitative_profiles])

        barchart = html.Div([
            html.Div([html.H5('Grouped sentiment counts by data attributes', style={
                'textAlign': 'center', 'marginTop': 20, 'marginBottom': 20,
                # 'color': colours['text']
            })], style={'marginTop': 5, 'marginBottom': 5}),
            html.Div([
                html.Div(
                    [
                        dcc.Dropdown(
                            id="Attribute",
                            options=options,
                            value='None'),
                    ],
                    style={'width': '25%',
                           'display': 'inline-block'}, className='six columns'),
                html.Div([dcc.RadioItems(
                    id='radioBarchart',
                    options=[
                        {'label': '  Raw counts', 'value': 'raw'},
                        {'label': '  Normalised counts', 'value': 'normalised'}
                    ],
                    value='raw'
                )], className='six columns')
            ], className='row', style={'align': 'center', 'marginLeft': 20}),
            dcc.Graph(id='barchart'),
        ])
        return barchart

    def make_boxplot(self):
        boxplot = html.Div([
            html.Div([html.H5('Quantitative attributes by sentiment', style={
                'textAlign': 'center',
                # 'color': colours['text']
            })], style={'marginTop': 20, 'marginBottom': 20}),
            html.Div(
                [
                    dcc.Dropdown(
                        id="BoxAttribute",
                        options=[{
                            'label': i,
                            'value': i
                        } for i in self.quantitative_profiles],
                        value='None'),
                ],
                style={'width': '25%',
                       'display': 'inline-block'}),
            dcc.Graph(id='boxplot'),
        ])
        return boxplot

    def make_timeseries(self):
        timeseries = html.Div([
            html.H5('Number of reviews per day', style={
                'textAlign': 'center',
                # 'color': colours['text']
            }),
            html.Div([dcc.RadioItems(
                id='radioTimeseries',
                options=[
                    {'label': '  Raw counts', 'value': 'raw'},
                    {'label': '  Normalised counts', 'value': 'normalised'}
                ],
                value='raw'
            )], style={'marginLeft': 20}),
            dcc.Graph(id='timeseries')
        ])
        return timeseries

    def make_tree(self):
        tree = html.Div([html.H5('Classification tree analysis',
                                 style={
                                     'textAlign': 'center',
                                     # 'color': colours['text'],
                                     'marginBottom': 25, 'marginTop': 25}
                                 ),

                         html.Div([
                             html.Div(html.P("Splitting criterion"), className="six columns",
                                      style={'textAlign': 'right'}),
                             html.Div(dcc.Dropdown(
                                 options=[
                                     {'label': 'Gini impurity index', 'value': 'gini'},
                                     {'label': 'Information gain', 'value': 'entropy'},
                                 ], id='crit', value='gini'), className="six columns", style={'maxWidth': '500px',
                                                                                              }),
                         ], className='row', style={'marginBottom': 5, 'marginTop': 5}),
                         html.Div([
                             html.Div(html.P("Splitting strategy"), className="six columns",
                                      style={'textAlign': 'right'}),
                             html.Div(dcc.Dropdown(
                                 options=[
                                     {'label': 'Best split', 'value': 'best'},
                                     {'label': 'Best random split', 'value': 'random'},
                                 ], id='split', value='best'), className="six columns", style={'maxWidth': '500px',
                                                                                               }),
                         ], className='row', style={'marginBottom': 5, 'marginTop': 5}),
                         html.Div([
                             html.Div(html.P("Class weights"), className="six columns",
                                      style={'textAlign': 'right'}),
                             html.Div(dcc.Dropdown(
                                 options=[
                                     {'label': 'Balanced', 'value': 'balanced'},
                                     {'label': 'Not balanced', 'value': None},
                                 ], id='class_weights', value='balanced'),
                                 className="six columns", style={'maxWidth': '500px'})
                         ], className='row', style={'marginBottom': 5, 'marginTop': 5}),

                         html.Div([
                             html.Div(html.P("Max depth", className="six columns"), style={'textAlign': 'right'}),
                             html.Div(dcc.Input(
                                 placeholder='Enter a value...',
                                 type='number',
                                 value=5,
                                 id='max_depth'), className="six columns", style={'textAlign': 'left',
                                                                                  }),
                         ], className="row", style={'marginBottom': 5, 'marginTop': 5}),

                         html.Div([
                             html.Div(html.P("Minimum number/fraction of observations to warrant split",
                                             className="six columns"), style={'textAlign': 'right'}),
                             html.Div(dcc.Input(
                                 placeholder='Enter a value...',
                                 type='number',
                                 value=0.1,
                                 step=0.05,
                                 id='min_split'
                             ), className="six columns", style={'textAlign': 'left',
                                                                }),
                         ], className='row', style={'marginBottom': 5, 'marginTop': 5}),
                         html.Div([
                             html.Div(html.P("Minimum number/fraction of observations in leaf node",
                                             className="six columns"),
                                      style={'textAlign': 'right'}),
                             html.Div(dcc.Input(
                                 placeholder='Enter a value...',
                                 type='number',
                                 value=0.3,
                                 step=0.05,
                                 id='min_leaf'), className="six columns", style={'textAlign': 'left',
                                                                                 }),
                         ], className='row', style={'marginBottom': 5, 'marginTop': 5}),
                         html.Div([
                             html.Div(html.P("Max number/fraction of features to consider during split",
                                             className="six columns", style={'textAlign': 'right'})),
                             html.Div(dcc.Input(
                                 placeholder='Enter a value...',
                                 type='number',
                                 value=0.8,
                                 step=0.05,
                                 id='max_features'), className="six columns", style={'textAlign': 'left',
                                                                                     }),
                         ], className="row", style={'marginBottom': 5, 'marginTop': 5}),

                         html.Div(html.Button('Display tree', id='tree_button'), style={'align': 'center',
                                                                                        'text-align': 'center',
                                                                                        'marginTop': 25,
                                                                                        'marginBottom': 25}),
                         html.Div(id='tree_performance', style={'text-align': 'center', 'marginTop': 25,
                                                                'marginBottom': 25}),
                         html.Div([
                             html.P("For categorical features interpret 'X_Y < 0.5 is False' "
                                    "to mean feature X = Y")],
                                  style={'text-align': 'center', 'marginTop': 25,
                                                                'marginBottom': 25}),
                         html.Div(html.Img(id='tree_img', style={'maxWidth': '1300px'}),
                                  style={'text-align': 'center', 'display': 'inline-block',
                                         'width': '600px'})
                         ])
        return tree

    def make_geomap(self):
        df = self.unified.copy()
        sentiments = ['negative', 'neutral', 'positive']

        df = df.assign(q=np.ones(df.shape[0]))
        pv = pd.pivot_table(
            df,
            columns=["Sentiment"],
            index=['Latitude', 'Longitude'],
            values=['q'],
            aggfunc='sum',
            fill_value=0)

        names = df.groupby(['Latitude', 'Longitude']).first()[self.locations[2]]

        totals = np.asarray(pv[('q', 'positive')].values, dtype='int64') + np.asarray(pv[('q', 'negative')].values,
                                                                                      dtype='int64') + np.asarray(
            pv[('q', 'neutral')].values, dtype='int64')
        max = np.max(totals)
        self.multiplier = 90 / max

        cases = []
        for i in sentiments:
            cases.append(go.Scattermapbox(
                lon=pv[('q', i)].index.get_level_values(1),
                lat=pv[('q', i)].index.get_level_values(0),
                text=pv[('q', i)].values,
                hoverinfo='text',
                hovertext=pv[('q', i)].values,  # pv[('q', i)].values.astype(str),
                name=i,
                marker=go.scattermapbox.Marker(
                    size=pv[('q', i)].values * self.multiplier,
                    color=colours['sentiment_dict'][i],
                    opacity=0.8,
                )
            ))

        cases.append(go.Scattermapbox(
            lon=pv[('q', i)].index.get_level_values(1),
            lat=pv[('q', i)].index.get_level_values(0),
            text=names.astype(str) + " " + totals.astype(str),
            name='Location names',
            visible='legendonly',
            hoverinfo='text',
            mode="markers+text",
            textposition='bottom center',
            marker=go.scattermapbox.Marker(
                size=0.1,
                color="#000000",
            )
        ))

        layout = go.Layout(
            width=1400,
            height=800,
            hovermode="closest",
            title=go.layout.Title(
                text='Sentiment counts by geographical location'),
            mapbox=go.layout.Mapbox(
                accesstoken='pk.eyJ1IjoiamFjcXVlbGluZWthem1haWVyIiwiYSI6ImNqd2JyNm1uODBsdTM0M3M2YnF3Z3U0cDkifQ.HkemCJfHUH7nNHOmb21q0g',
                bearing=0,
                center=go.layout.mapbox.Center(
                    lat=-30,
                    lon=20
                ),
                pitch=0,
                zoom=4,
                style='light'
            ),
            legend=go.layout.Legend(
                traceorder='reversed', x=0,y=1
            )
        )

        fig= {'layout': layout, 'data': cases}

        map = html.Div([dcc.Graph(id='geomap',config={'scrollZoom': True}, figure=fig)],
                       style={'textAlign':'center'},
                       )

        return map

    def make_topic_analysis(self):
        df = self.unified

        # Noun phrase extraction
        nouns = []
        # function to test if something is a noun
        is_noun = lambda pos: pos[:2] == 'NN'

        for message in df['Review'].str.lower():
            tokenized = [i for i in nltk.word_tokenize(message) if i not in nltk.corpus.stopwords.words('english')]
            local_nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
            if len(local_nouns)>0:
                nouns.extend(local_nouns)

        c = collections.Counter(nouns)
        keys = [key for key, val in c.most_common(30)]
        options= []
        options.extend([{
            'label': i,
            'value': i
        } for i in keys])

        # Keyword displays
        display = html.Div([
                html.Div([html.H5('Sentiment counts by keywords', style={
                'textAlign': 'center', 'marginTop': 20, 'marginBottom': 20,
                # 'color': colours['text']
                })], style={'marginTop': 5, 'marginBottom': 5}),
                html.Div([
                    html.Div([dcc.RadioItems(
                        id='radioTopics',
                        options=[
                            {'label': '  Raw counts', 'value': 'raw'},
                            {'label': '  Normalised counts', 'value': 'normalised'}
                        ],
                        value='raw'
                    )],className='six columns'),
                    html.Div([dcc.Dropdown(id='topic_graph_Ddl', multi=True, placeholder="Select keywords to compare")],
                             style = {'textAlign':'center','maxWidth':600}, className='six columns')
                ], className='row'),
                dcc.Graph(id='topic_graph'),
        ])

        # Topic modelling
        layout = html.Div([
            html.Div(id='placeholder', style={'display': 'none'}),
            html.Div([html.H5('Noun phrase detection')], style = {'textAlign':'center', 'padding':10}),
            html.Div([
                html.Div([
                    html.P('Select automatically detected noun phrases for filter: ')], style={'textAlign':'right'},
                className='six columns'),
                html.Div([
                    dcc.Dropdown(
                        options = options,
                        multi=True, id = 'keywords'
                        )
                ], className='six columns', style={'padding':5})
            ], className='row', style = {'textAlign':'center', 'padding':5}),
            html.Div([
                html.Div([
                    html.P('Add keywords to list: ')], style = {'textAlign': 'right'}
            , className = 'six columns'),
            html.Div([
                dcc.Input(
                    placeholder='Enter a keyword...',
                    type='text',
                    value='', id='manual_keywords_input'
                ),
                html.Button('Add', id='manual_keywords_button')
            ], className='six columns')
            ], className = 'row'),
            html.Div([
                html.Div([html.H5('LDA Topic modelling')], style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.P('Number of topics:')], style={'textAlign': 'right'}
                        , className='six columns'),
                    html.Div([
                        dcc.Input(
                            placeholder='Enter a number...',
                            type='number',
                            value=None, id='num_topics'
                        )
                    ], className='six columns')
                ], className='row', style={'marginTop':5,'marginBottom':5}),\

                html.Div([
                    html.Div([
                        html.P('Number of iterations: ')], style={'textAlign': 'right'}
                        , className='six columns'),
                    html.Div([
                        dcc.Input(
                            placeholder='Enter a number...',
                            type='number',
                            value='', id='num_passes'
                        )
                    ], className='six columns')
                ], className='row', style={'marginTop':5, 'marginBottom':5}),

                html.Div([html.Button('Execute and view topic model',
                                      id='lda_button')], style={'padding':10, 'textAlign':'center'})
            ], style={'marginTop':20}),
            display
        ])
        return layout

    ##################################################
    # Deploy app
    ##################################################
    def deploy(self):
        dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        dash_app.scripts.config.serve_locally = True
        dash_app.css.append_css({
            'external_url': 'https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css'})

        ##############################
        # Create layout
        ##############################
        dash_app.layout = html.Div(children=[
            html.Div([html.H1('Analyse results',
                    style={
                        'textAlign': 'center',
                    })], style={'marginTop': 10}),
            html.Div([
                html.P('Filter results by keywords. Customise list in Topic Analysis tab.'),
                dcc.Dropdown(
                    id='keywordsDdl',multi=True,
                    options=[
                        {'label': 'No filter', 'value': 'All'},
                        {'label': 'ATM', 'value': 'atm'},
                        {'label': 'Service', 'value': 'service'},
                        {'label': 'Loan', 'value': 'loan'},
                        {'label': 'App', 'value': 'app'}
                    ],
                    placeholder="Select keywords to filter on",
                )
            ], style={'padding': 20}),
            html.Div([
                dcc.Tabs(id="tabs", children=[

                    dcc.Tab(label='Summary view', children=[
                        html.Div([
                            html.Div([html.Div([self.make_piechart()], className='six columns'),
                                      html.Div([html.H5("Summary"),
                                                html.P(id='summary')
                                                ], className='six columns', style={'textAlign': 'justify',
                                                                                   'maxWidth':500, 'marginTop':10}
                                               )], className='row'),
                            html.H5("Sample documents",
                                    style={'marginTop': 5, 'textAlign': 'center'}),
                            html.Div([
                                html.Div([
                                    html.P("Randomly selected negative sample",
                                           style={'marginTop': 5, 'width': '400px'}),
                                    dcc.Textarea(
                                        placeholder='Click to select random sample message',
                                        disabled=True, style={'width': '400px'},
                                        id='negReview'
                                    ),
                                ], className='four columns'),
                                html.Div([
                                    html.P("Randomly selected neutral sample", style={'marginTop': 5, 'marginLeft': 5}),
                                    dcc.Textarea(
                                        placeholder='Click to select random sample message',
                                        disabled=True, style={'width': '400px'},
                                        id='neutReview'
                                    ),
                                ], className='four columns'),
                                html.Div([
                                    html.P("Randomly selected positive sample", style={'marginTop': 5}),
                                    dcc.Textarea(
                                        placeholder='Click to select random sample message',
                                        disabled=True, style={'width': '400px'},
                                        id='posReview'
                                    ),
                                ], className='four columns'),
                            ], className='row'),
                            html.Div([
                                html.Button("Randomise samples", id='posButton')], style={'textAlign': 'center'}),
                            html.H5("Visualisation of frequent terms in corpus",
                                    style={'marginTop': 20, 'marginBottom': 10, 'textAlign': 'center'}),
                            html.Div([
                                html.Div([
                                    html.P("Frequent terms in negative documents",
                                           style={'marginTop': 5, 'width': '400px'}),
                                    html.Img(id='negWordCloud', style={'width': '400px'})
                                ], className='four columns'),
                                html.Div([
                                    html.P("Frequent terms in neutral documents", style={'marginTop': 5}),
                                    html.Img(id='neutWordCloud', style={'width': '400px'})
                                ], className='four columns'),
                                html.Div([
                                    html.P("Frequent terms in positive documents", style={'marginTop': 5}),
                                    html.Img(id='posWordCloud', style={'width': '400px'})
                                ], className='four columns'),
                            ], className='row'),

                            html.Div([
                                html.Div([
                                    html.P('Remove certain words from wordcloud, separated by commas',
                                           style={'textAlign': 'right'})
                                ], className='two columns', style={'width': '500px'}),
                                html.Div([
                                    dcc.Input(
                                        placeholder='Enter words here',
                                        type='text',
                                        value='',
                                        id="cloud_stopwords",
                                        style={'width': '500px'}
                                    ),
                                ], className='two columns'),
                            ], style={'textAlign': 'center', 'marginTop': 5, 'marginBottom': 10}, className='row')

                        ], style={'marginLeft': 5})
                    ]),

                    dcc.Tab(label='Topic Analysis', children=[
                        self.make_topic_analysis()
                    ]),

                    dcc.Tab(label='Basic visualisations', children=[
                        html.Div(id='df-tab1', style={'display': 'none'}),
                        html.Div([
                            html.Div([
                                self.make_barchart()
                            ], className="six columns"),
                            html.Div([
                                self.make_boxplot()
                            ], className="six columns")
                        ], className="row"),
                        self.make_timeseries()
                    ]),

                    dcc.Tab(label='Map view', children=[
                        self.make_geomap()
                    ]),

                    dcc.Tab(label='Multivariate Analysis', children=[
                        self.make_tree(),
                    ])
                ])
            ])
        ],)

        ##############################
        # Make interactive
        ##############################
        # Filter on dropdown keywords
        @dash_app.callback(Output('df-tab1', 'children'), [Input('keywordsDdl', 'value')])
        def filter_data(keywords):
            # filter only reviews that contain keywords
            df = self.unified
            if keywords != ['All'] and keywords != None:
                reviews = self.sel_tokens
                for word in keywords:
                    df = df[list(map(lambda x:word in x, reviews))]
            print(df.info())
            print(df.describe())
            filtered_df = df.to_json(orient='split', date_format='iso')

            return json.dumps(filtered_df)

        # Overview page
        @dash_app.callback([Output('summary', 'children'), Output('negReview', 'value'), Output('negWordCloud', 'src'),
                            Output('posReview', 'value'), Output('posWordCloud', 'src'),
                            Output('neutReview', 'value'), Output('neutWordCloud', 'src')],
                           [Input('df-tab1', 'children'), Input('posButton', 'n_clicks')],
                            [State('cloud_stopwords', 'value')]
                           )
        def update_overview(filtered_df, n_clicks, input_stopwords):
            if filtered_df is not None:
                df = json.loads(filtered_df)
                df = pd.read_json(df, orient='split')
            else:
                df = self.unified

            cloud_stopwords = stopwords.words('english')
            cloud_stopwords.extend(input_stopwords.split(","))

            pos, neg, neut = self.count_instances(df['Sentiment'])
            # Noun phrase extraction
            nouns_sorted = {}
            # function to test if something is a noun
            is_noun = lambda pos: pos[:2] == 'NN'

            for sentiment in ['positive', 'negative', 'neutral']:
                nouns = []
                for message in df[df['Sentiment'] == sentiment]['Review'].str.lower():
                    tokenized = [i for i in nltk.word_tokenize(message) if i not in nltk.corpus.stopwords.words('english')]
                    local_nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
                    if len(local_nouns) > 0:
                        nouns.extend(local_nouns)
                c = collections.Counter(nouns)
                keys = [key for key, val in c.most_common(3)]
                nouns_sorted[sentiment] = keys

            if pos > 0 and neg > 0:
                summary = "The current selection comprises {} documents, of which {} were classified as positive, " \
                      "{} were classified as negative and {} were classified as neutral by the selected model. "\
                      "Positive documents typically mentioned '{}', '{}' and '{}'. "\
                      "Salient terms in negative reviews include '{}', '{}' and '{}'. ".format(
                df.shape[0], pos, neg, neut,
                nouns_sorted['positive'][0], nouns_sorted['positive'][1],nouns_sorted['positive'][2],
                nouns_sorted['negative'][0], nouns_sorted['negative'][1], nouns_sorted['negative'][2],
             )
            elif neg > 0:
                summary = "The current selection comprises {} documents, of which {} were classified as positive, " \
                          "{} were classified as negative and {} were classified as neutral by the selected model. " \
                          "Negative documents typically mentioned '{}', '{}' and '{}'. " .format(
                    df.shape[0], pos, neg, neut,
                    nouns_sorted['negative'][0], nouns_sorted['negative'][1], nouns_sorted['negative'][2]
                )
            elif pos > 0:
                summary = "The current selection comprises {} documents, of which {} were classified as positive, " \
                          "{} were classified as negative and {} were classified as neutral by the selected model. " \
                          "Positive documents typically mentioned '{}', '{}' and '{}'. ".format(
                    df.shape[0], pos, neg, neut,
                    nouns_sorted['positive'][0], nouns_sorted['positive'][1], nouns_sorted['positive'][2]
                )
            else:
                summary = "The current selection comprises {} documents, of which {} were classified as positive, " \
                          "{} were classified as negative and {} were classified as neutral by the selected model. ".format(
                    df.shape[0], pos, neg, neut
                )


            if neg > 0:
                index_neg = random.randint(0, neg - 1)
                neg_msg = df[df['Sentiment'] == 'negative']['Review'].iloc[index_neg]

                all_neg = " ".join(review for review in df[df['Sentiment'] == 'negative']['Review'])
                neg_path = 'assets/neg_wordcloud.png'
                encoded_neg_cloud = self.create_wordcloud(all_neg, neg_path, cloud_stopwords)
                decoded_neg_cloud = 'data:image/png;base64,{}'.format(encoded_neg_cloud.decode())
            else:
                neg_msg = "No negative messages in selection."
                decoded_neg_cloud = ""

            if pos > 0:
                index_pos = random.randint(0, pos - 1)
                pos_msg = df[df['Sentiment'] == 'positive']['Review'].iloc[index_pos]

                all_pos = " ".join(review for review in df[df['Sentiment'] == 'positive']['Review'])
                pos_path = 'assets/pos_wordcloud.png'
                encoded_pos_cloud = self.create_wordcloud(all_pos, pos_path, cloud_stopwords)
                decoded_pos_cloud = 'data:image/png;base64,{}'.format(encoded_pos_cloud.decode())
            else:
                pos_msg = "No positive messages in selection."
                decoded_pos_cloud = ""

            if neut > 0:
                index_neut = random.randint(0, neut - 1)
                neut_msg = df[df['Sentiment'] == 'neutral']['Review'].iloc[index_neut]

                all_neut = " ".join(review for review in df[df['Sentiment'] == 'neutral']['Review'])
                neut_path = 'assets/neut_wordcloud.png'
                encoded_neut_cloud = self.create_wordcloud(all_neut, neut_path, cloud_stopwords)
                decoded_neut_cloud = 'data:image/png;base64,{}'.format(encoded_neut_cloud.decode())
            else:
                neut_msg = "No neutral messages in selection."
                decoded_neut_cloud = ""

            return summary, neg_msg, decoded_neg_cloud, pos_msg, decoded_pos_cloud, neut_msg, decoded_neut_cloud

        # Pie chart
        @dash_app.callback(Output('piechart', 'figure'), [Input('df-tab1', 'children')])
        def update_piechart(filtered_df):
            if filtered_df is not None:
                df = json.loads(filtered_df)
                df = pd.read_json(df, orient='split')
            else:
                df = self.unified

            labels = ['positive', 'negative', 'neutral']
            values = self.count_instances(df['Sentiment'])

            figure = go.Figure(data=[go.Pie(labels=labels, values=values,
                                            hoverinfo='label+value', textinfo='percent',
                                            marker=dict(colors=colours['sentiments']), opacity=0.8)],
                               layout=go.Layout(title='Overall sentiment')
                               )
            return figure

        # Barchart
        @dash_app.callback(
            Output('barchart', 'figure'),
            [Input('Attribute', 'value'), Input('df-tab1', 'children'), Input('radioBarchart', 'value')])
        def update_barchart(Attribute, filtered_df, type):
            if filtered_df is not None:
                df = json.loads(filtered_df)
                df = pd.read_json(df, orient='split')
            else:
                df = self.unified

            if Attribute == 'None':
                # display regular barchart
                if type == 'raw':
                    trace1 = go.Bar(x=['All'], y=[sum(df['Sentiment'] == 'positive')], name='positive',
                                    marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                    trace2 = go.Bar(x=['All'], y=[sum(df['Sentiment'] == 'negative')], name='negative',
                                    marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                    trace3 = go.Bar(x=['All'], y=[sum(df['Sentiment'] == 'neutral')], name='neutral',
                                    marker=dict(color=colours['sentiments'][2]), opacity=0.8)
                else:
                    total = sum(df['Sentiment'] == 'positive') + sum(df['Sentiment'] == 'negative') + sum(
                        df['Sentiment'] == 'neutral')
                    trace1 = go.Bar(x=['All'], y=[sum(df['Sentiment'] == 'positive') / total], name='positive',
                                    marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                    trace2 = go.Bar(x=['All'], y=[sum(df['Sentiment'] == 'negative') / total], name='negative',
                                    marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                    trace3 = go.Bar(x=['All'], y=[sum(df['Sentiment'] == 'neutral') / total], name='neutral',
                                    marker=dict(color=colours['sentiments'][2]), opacity=0.8)

                return go.Figure({'data': [trace1, trace2, trace3],
                                  'layout': go.Layout(
                                      title='Total sentiment counts',
                                      barmode='group')})
            else:
                df = df.assign(q=np.ones(df.shape[0]))
                pv = pd.pivot_table(
                    df,
                    index=[Attribute],
                    columns=["Sentiment"],
                    values=['q'],
                    aggfunc='sum',
                    fill_value=0)

                if type == 'normalised':
                    sums = []
                    for name in pv.index:
                        localsum = pv[('q', 'positive')][name] + pv[('q', 'negative')][name] + pv[('q', 'neutral')][
                            name]
                        sums.append(localsum)

                    try:
                        trace1 = go.Bar(x=pv.index, y=pv[('q', 'positive')] / sums, name='positive',
                                        marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                    except:
                        trace1 = go.Bar(x=pv.index, y=[0] / sums, name='positive',
                                        marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                    try:
                        trace2 = go.Bar(x=pv.index, y=pv[('q', 'negative')] / sums, name='negative',
                                        marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                    except:
                        trace2 = go.Bar(x=pv.index, y=[0] / sums, name='negative',
                                        marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                    try:
                        trace3 = go.Bar(x=pv.index, y=pv[('q', 'neutral')] / sums, name='neutral',
                                        marker=dict(color=colours['sentiments'][2]), opacity=0.8)
                    except:
                        trace3 = go.Bar(x=pv.index, y=[0] / sums, name='neutral',
                                        marker=dict(color=colours['sentiments'][2]), opacity=0.8)

                else:  # if type == raw

                    try:
                        trace1 = go.Bar(x=pv.index, y=pv[('q', 'positive')], name='positive',
                                        marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                    except:
                        trace1 = go.Bar(x=pv.index, y=[0], name='positive',
                                        marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                    try:
                        trace2 = go.Bar(x=pv.index, y=pv[('q', 'negative')], name='negative',
                                        marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                    except:
                        trace2 = go.Bar(x=pv.index, y=[0], name='negative',
                                        marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                    try:
                        trace3 = go.Bar(x=pv.index, y=pv[('q', 'neutral')], name='neutral',
                                        marker=dict(color=colours['sentiments'][2]), opacity=0.8)
                    except:
                        trace3 = go.Bar(x=pv.index, y=[0], name='neutral',
                                        marker=dict(color=colours['sentiments'][2]), opacity=0.8)

                return go.Figure({'data': [trace1, trace2, trace3],
                                  'layout': go.Layout(
                                      title='Sentiment counts by {}'.format(Attribute),
                                      barmode='stack')})

        # Boxplot
        @dash_app.callback(
            Output('boxplot', 'figure'),
           [Input('BoxAttribute', 'value'), Input('df-tab1', 'children')])
        def update_boxplot(Attribute, filtered_df):
            if filtered_df is not None:
                df = json.loads(filtered_df)
                df = pd.read_json(df, orient='split')
            else:
                df = self.unified

            if Attribute == 'None':
                # display regular boxplot? noo
                pass
            else:
                trace1 = go.Box(y=df[df['Sentiment'] == 'positive'][Attribute], name='positive',
                                marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                trace2 = go.Box(y=df[df['Sentiment'] == 'negative'][Attribute], name='negative',
                                marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                trace3 = go.Box(y=df[df['Sentiment'] == 'neutral'][Attribute], name='neutral',
                                marker=dict(color=colours['sentiments'][2]), opacity=0.8)

                return go.Figure({
                    'data': [trace1, trace2, trace3],
                    'layout':
                        go.Layout(
                            title='{} per sentiment polarity class'.format(Attribute),
                        )
                })

        # Time series
        @dash_app.callback(
            Output('timeseries', 'figure'),
            [Input('df-tab1', 'children'), Input('radioTimeseries', 'value')]
        )
        def update_timeseries(filtered_df, type):
            if filtered_df is not None:
                df = json.loads(filtered_df)
                df = pd.read_json(df, orient='split')
            else:
                df = self.unified

            # get sentiment counts by unique dates
            df = df.assign(q=np.ones(df.shape[0]))
            pv = pd.pivot_table(
                df,
                index=[self.date],
                columns=["Sentiment"],
                values=['q'],
                aggfunc='sum',
                fill_value=0)

            if type == 'normalised':
                sums = []
                for name in pv.index:
                    localsum = pv[('q', 'positive')][name] + pv[('q', 'negative')][name] + pv[('q', 'neutral')][name]
                    sums.append(localsum)
            else:
                sums = np.ones(len(pv.index))

            try:
                trace_pos = go.Scatter(x=pv.index, y=pv[('q', 'positive')] / sums, name='Number positive reviews',
                                       marker=dict(color=colours['sentiments'][0]), opacity=0.8)
            except:
                trace_pos = go.Scatter(x=pv.index, y=[0], name='Number positive reviews',
                                       marker=dict(color=colours['sentiments'][0]), opacity=0.8)
            try:
                trace_neg = go.Scatter(x=pv.index, y=pv[('q', 'negative')] / sums, name='Number negative reviews',
                                       marker=dict(color=colours['sentiments'][1]), opacity=0.8)
            except:
                trace_neg = go.Scatter(x=pv.index, y=[0], name='Number negative reviews',
                                       marker=dict(color=colours['sentiments'][1]), opacity=0.8)
            try:
                trace_neut = go.Scatter(x=pv.index, y=pv[('q', 'neutral')] / sums, name='Number neutral reviews',
                                        marker=dict(color=colours['sentiments'][2]), opacity=0.8)
            except:
                trace_neut = go.Scatter(x=pv.index, y=[0], name='Number neutral reviews',
                                        marker=dict(color=colours['sentiments'][2]), opacity=0.8)

            data = [trace_pos, trace_neg, trace_neut]

            layout = dict(
                xaxis=dict(
                    range=[df[self.date].min(), df[self.date].max()],
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7,
                                 label='1w',
                                 step='day',
                                 stepmode='backward'),
                            dict(count=1,
                                 label='1m',
                                 step='month',
                                 stepmode='backward'),
                            dict(count=6,
                                 label='6m',
                                 step='month',
                                 stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider={'visible': True, 'range': [df[self.date].min(), df[self.date].max()],
                                 'autorange': False, },
                    type='date'
                ))

            fig = dict(data=data, layout=layout)
            return go.Figure(fig)

        # Decision tree
        @dash_app.callback(
            [Output('tree_img', 'src'), Output('tree_performance', 'children')],
            [Input('df-tab1', 'children'),
             Input('tree_button', 'n_clicks'), Input('min_split', 'value'),
             Input('min_leaf', 'value'), Input('max_features', 'value'),
             Input('crit', 'value'), Input('split', 'value'),
             Input('class_weights', 'value'), Input('max_depth', 'value')]
        )
        def update_tree(filtered_df, n_clicks, min_split, min_leaf, max_features, crit, split, weights, depth):
            if filtered_df is not None:
                df = json.loads(filtered_df)
                df = pd.read_json(df, orient='split')
            else:
                df = self.unified
            estimator = DecisionTreeClassifier(random_state=0
                                               , criterion=crit
                                               , splitter=split
                                               , max_depth=depth
                                               , min_samples_split=min_split
                                               , min_samples_leaf=min_leaf
                                               , class_weight=weights
                                               , max_features=max_features
                                               )

            X = df.iloc[:, 3:]  # all but name, review and sentiment
            selected = self.quantitative_profiles
            selected.extend(self.qualitative_profiles)
            X = X[selected]
            X_transformed = pd.get_dummies(X, columns=self.qualitative_profiles, drop_first=True)

            y = df.iloc[:, 2]

            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2)
            estimator.fit(X_train, y_train)
            score = "Estimated model accuracy: %.3f" % estimator.score(X_test, y_test)
            estimator.fit(X_transformed, y)

            dot_data = StringIO()
            export_graphviz(estimator, out_file=dot_data,
                            feature_names=X_transformed.columns,
                            class_names=estimator.classes_,
                            filled=True, rounded=True,
                            special_characters=True
                            )
            try:
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_png("assets/dtree.png")
            except:
                (graph,) = pydotplus.graph_from_dot_data(dot_data.getvalue())
                graph.write_png("assets/dtree.png")


            encoded_image = base64.b64encode(open("assets/dtree.png", 'rb').read())

            return 'data:image/png;base64,{}'.format(encoded_image.decode()), html.P(score)

        # Geomap
        @dash_app.callback(
            Output('geomap', 'figure'),
            [Input('df-tab1', 'children')]
        )
        def update_map(filtered_df):
            if filtered_df is not None:
                df = json.loads(filtered_df)
                df = pd.read_json(df, orient='split')
            else:
                df = self.unified.copy()

            sentiments = []
            pos, neg, neut = self.count_instances(df['Sentiment'])
            if neg > 0:
                sentiments.append('negative')
            if neut > 0:
                sentiments.append('neutral')
            if pos > 0:
                sentiments.append('positive')

            df = df.assign(q=np.ones(df.shape[0]))
            pv = pd.pivot_table(
                df,
                columns=["Sentiment"],
                index=['Latitude', 'Longitude'],
                values=['q'],
                aggfunc='sum',
                fill_value=0)

            names = df.groupby(['Latitude', 'Longitude']).first()[self.locations[2]]

            totals = 0
            for i in sentiments:
                totals += np.asarray(pv[('q', i)].values, dtype='int64')

            cases = []
            for i in sentiments:
                cases.append(go.Scattermapbox(
                    lon=pv[('q', i)].index.get_level_values(1),
                    lat=pv[('q', i)].index.get_level_values(0),
                    text=pv[('q', i)].values,
                    hoverinfo='name+text',
                    hovertext= names+": "+pv[('q', i)].values.astype(str),
                    name=i,
                    marker=go.scattermapbox.Marker(
                        size=pv[('q', i)].values * self.multiplier,
                        color=colours['sentiment_dict'][i],
                        opacity = 0.8,
                    )
                ))


            cases.append(go.Scattermapbox(
                lon=pv[('q', i)].index.get_level_values(1),
                lat=pv[('q', i)].index.get_level_values(0),
                text=names.astype(str) + " " + totals.astype(str),
                name='Location names',
                visible='legendonly',
                hoverinfo = 'text',
                mode = "markers+text",
                textposition = 'bottom center',
                marker=go.scattermapbox.Marker(
                    size=0.1,
                    color="#000000",
                )
            ))

            layout = go.Layout(
                width=1400,
                height=800,
                hovermode="closest",
                title=go.layout.Title(
                   text='Sentiment counts by geographical location'),
                mapbox=go.layout.Mapbox(
                    accesstoken='pk.eyJ1IjoiamFjcXVlbGluZWthem1haWVyIiwiYSI6ImNqd2JyNm1uODBsdTM0M3M2YnF3Z3U0cDkifQ.HkemCJfHUH7nNHOmb21q0g',
                    bearing=0,
                    center=go.layout.mapbox.Center(
                        lat=-30,
                        lon=20
                    ),
                    pitch=0,
                    zoom=4,
                    style='light'
                ),
                legend=go.layout.Legend(
                    traceorder='reversed', x=0, y=1
                )
            )
            return {'layout':layout, 'data':cases}

        # Topic keywords
        @dash_app.callback(
            Output('keywordsDdl', 'options'),
            [Input('keywords', 'value')]
        )
        def update_keywordslist(keywords):
            options=[]
            options.append({'label': 'No filter', 'value':'All'})
            try:
                options.extend([{
                    'label':i,
                    'value': i
                } for i in keywords])
            except:
                pass
            return options

        @dash_app.callback([Output('keywords', 'options'), Output('manual_keywords_input', 'value'),
                            Output('topic_graph_Ddl', 'options')],
                           [Input('manual_keywords_button', 'n_clicks')],
                           [State('manual_keywords_input', 'value'), State('keywords', 'options')],
                    )
        def add_keyword_entry(n_clicks, new, old):
            if new != "":
                if new != " ":
                    old.extend([{'label': new, 'value': new}])

            return old, '', old

        @dash_app.callback(Output('topic_graph', 'figure'),
                           [Input('topic_graph_Ddl', 'value'), Input('radioTopics', 'value')]
                           )
        def display_topicgraph(topics, type):
            df = self.unified
            reviews = self.sel_tokens

            x = []
            y_pos = []
            y_neg = []
            y_neut = []

            if topics is not None:
                sums=[]
                for topic in topics:
                    dff = df[list(map(lambda x: topic in x, reviews))]
                    x.append(topic)
                    y_pos.append(sum(dff['Sentiment'] == 'positive'))
                    y_neg.append(sum(dff['Sentiment'] == 'negative'))
                    y_neut.append(sum(dff['Sentiment'] == 'neutral'))

                    if type == 'normalised':
                        localsum = sum(dff['Sentiment'] == 'positive') \
                                    + sum(dff['Sentiment'] == 'negative') \
                                    + sum(dff['Sentiment'] == 'neutral')
                        sums.append(localsum)
                    else:
                        sums = np.ones(len(topics))

                y_pos = np.array(y_pos)
                y_neg = np.array(y_neg)
                y_neut = np.array(y_neut)

                trace1 = go.Bar(x=x, y=y_pos/sums, name='positive',
                                marker=dict(color=colours['sentiments'][0]), opacity=0.8)
                trace2 = go.Bar(x=x, y=y_neg/sums, name='negative',
                                marker=dict(color=colours['sentiments'][1]), opacity=0.8)
                trace3 = go.Bar(x=x, y=y_neut/sums, name='neutral',
                                marker=dict(color=colours['sentiments'][2]), opacity=0.8)

                if type == 'normalised':
                    return go.Figure({'data': [trace1, trace2, trace3],
                                        'layout': go.Layout(barmode='stack')})
                else:
                    return go.Figure({'data': [trace1, trace2, trace3],
                                      'layout': go.Layout(barmode='group')})

        @dash_app.callback(Output('placeholder', 'children'), [Input('lda_button', 'n_clicks')],
                           [State('num_topics', 'value'),
                            State('num_passes', 'value')]
                           )
        def launch_lda(n_clicks,num_topics, num_passes):
            if num_topics is not None: #numpasses has default value
                common_texts = self.reviews.tokens
                common_dictionary = Dictionary(common_texts)
                # Filter out words that occur less than 2 documents, or more than 30% of the documents.
                common_dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100)
                common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
                lda = LdaMulticore(common_corpus, num_topics=num_topics, passes=num_passes,
                                   id2word=common_dictionary)
                vis_data = gensimvis.prepare(lda, common_corpus, common_dictionary)
                pyLDAvis.save_html(vis_data, 'lda.html')
                pyLDAvis.show(vis_data)

            return html.P('I am invisible')

        ##############################
        # Run App
        ##############################
        port = 5000 + random.randint(0, 999)
        url = "http://127.0.0.1:{0}".format(port)

        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

        dash_app.run_server(port=port, debug=False)