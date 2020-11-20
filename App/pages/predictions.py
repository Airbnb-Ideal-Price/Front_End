# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from test import get_inputs
from tensorflow.keras.models import load_model
import tensorflow.keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder


df = pd.read_csv('train.csv')
result = load_model('keras_model.h5')

def wrangle(X):
  # make a copy
  X = X.copy()

  # encode "t" and "f" as 1's and 0's
  X['host_has_profile_pic'][X['host_has_profile_pic']=='t'] = 1
  X['host_has_profile_pic'][X['host_has_profile_pic']=='f'] = 0

  X['host_identity_verified'][X['host_identity_verified']=='t'] = 1
  X['host_identity_verified'][X['host_identity_verified']=='f'] = 0

  X['instant_bookable'][X['instant_bookable']=='t'] = 1
  X['instant_bookable'][X['instant_bookable']=='f'] = 0

  # Group some of the many property types together
  X['property_type'][X['property_type'].isin(['Boat','Tent','Castle','Yurt', 'Hut', 'Treehouse',
                                              'Chalet','Earth House','Tipi','Cave',
                                              'Train','Parking Space','Island','Casa particular',
                                              'Lighthouse', 'Vacation home', 'Serviced apartment'])] = 'Other'

  # columns with unusable variance
  unusable_variance = ['zipcode']

  # columns with high percentage of missing values
  high_nans = ['first_review','host_response_rate','last_review',
               'review_scores_rating','thumbnail_url']

  # categorical values with high cardinality
  # 'neighborhood' has 620 and 'thumbnail_url' has many thousands
  high_card = ['neighbourhood','thumbnail_url','name','amenities',
               'description', 'id','host_since']

  # Get the price and drop the log of price
  X['int_price'] = np.exp(X['log_price'])
  X = X.drop(['log_price'] + unusable_variance + high_nans + high_card, axis=1)

  # Remove the upper 1% outliers in price
  X = X[(X['int_price'] <= np.percentile(X['int_price'], 99.0))]

  return X

df = wrangle(df)

target = 'int_price'
y = df[target]

X = df.drop('int_price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=.20,
                                                    random_state=0)

# Imports from this application
from app import app

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Predictions

            This is a tool for predicting the price of an AirBnB,
            Answer these questions and you will be returned with an approximate price below.
            
            """
        ),

        html.Div('Please put in information and click submit'),
        html.Br(),


    ],
    md=4,
)

column2 = dbc.Col(

    [  
        html.Div(["City: "]),
        dcc.Dropdown(
        id='city-dropdown',

        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'District of Columbia', 'value': 'DC'},
            {'label': 'San Francisco', 'value': 'SF'},
            {'label': 'Los Angeles', 'value': 'LA'},
            {'label': 'Chicago', 'value': 'Chicago'},
            {'label': 'Boston', 'value': 'Boston'}],
        value='LA'),

    html.Br(),


    html.Div(["Latitude: ",
              dcc.Input(id='my-input', value='45.5', type='text')]),
    html.Br(),

    html.Div(["Longitude: ",
              dcc.Input(id='longitude_input', value='-73.5', type='text')]),
    html.Br(),

        html.Div(["Property Type: "]),
        dcc.Dropdown(
            id='property-type-dropdown',

        options=[
            {'label': 'Apartment', 'value': 'Apartment'},
            {'label': 'House', 'value': 'House'},
            {'label': 'Condominium', 'value': 'Condominium'},
            {'label': 'Townhouse', 'value': 'Townhouse'},
            {'label': 'Loft', 'value': 'Loft'},
            {'label': 'Other', 'value': 'Other'},
            {'label': 'Guesthouse', 'value': 'Guesthouse'},
            {'label': 'Bed & Breakfast', 'value': 'Bed & Breakfast'},
            {'label': 'Bungalo', 'value': 'Bungalo'},
            {'label': 'Villa', 'value': 'Villa'},
            {'label': 'Dorm', 'value': 'Dorm'},
            {'label': 'Guest suite', 'value': 'Guest suite'},
            {'label': 'Camper/RV', 'value': 'Camper/RV'},
            {'label': 'Timeshare', 'value': 'Timeshare'},
            {'label': 'Cabin', 'value': 'Cabin'},
            {'label': 'In-law', 'value': 'In-law'},
            {'label': 'Hostel', 'value': 'Hostel'},
            {'label': 'Boutique hotel', 'value': 'Boutique hotel'}],
        value='Apartment'),

        html.Br(),

        html.Div(["Room Type: "]),
        dcc.Dropdown(
        id='room_type_dropdown',

        options=[
            {'label': 'Entire Home/Apt', 'value': 'Entire home/apt'},
            {'label': 'Private room', 'value': 'Private room'},
            {'label': 'Shared room', 'value': 'Shared room'}],
        value='Entire home/apt'),

    html.Br(),

    html.Div(["Accommodates:  ",
              dcc.Input(id='accommodates_input', value='1', type='text')]),
    html.Br(),

    html.Div(["Bathrooms:  ",
              dcc.Input(id='bathrooms_input', value='1.0', type='text')]),
    html.Br(),

    html.Div(["Bed Type: "]),
    dcc.Dropdown(
        id='bed_types_dropdown',

        options=[
            {'label': 'Real Bed', 'value': 'Real Bed'},
            {'label': 'Futon', 'value': 'Futon'},
            {'label': 'Pull-out Sofa', 'value': 'Pull-out Sofa'},
            {'label': 'Airbed', 'value': 'Airbed'},
            {'label': 'Couch', 'value': 'Couch'},],
        value='Real Bed'),

    html.Br(),

    html.Div(["Cancellation Policy: "]),
    dcc.Dropdown(
        id='cancellation_policy_dropdown',

        options=[
            {'label': 'Strict', 'value': 'strict'},
            {'label': 'Flexible', 'value': 'flexible'},
            {'label': 'Moderate', 'value': 'moderate'},
            {'label': 'Super Strict (30 day warning)', 'value': 'super_strict_30'},
            {'label': 'Super Strict (60 day warning)', 'value': 'super_strict_60'},],
        value='strict'),

    html.Br(),

    html.Div(["Does the AirBnB have a cleaning fee?: "]),
    dcc.Dropdown(
        id='cleaning_fee_check',

        options=[
            {'label': 'Yes', 'value': 'True'},
            {'label': 'No', 'value': 'False'},],
        value='True'),

    html.Br(),

    html.Div(["Does the owner have a profile picture?: ",
    dcc.Dropdown(
        id='profile_pic_check',

        options=[
            {'label': 'Yes', 'value': '1'},
            {'label': 'No', 'value': '0'},],
        value='1')]),

    html.Br(),

    html.Div(["Is the owner's identity verified?: ",
    dcc.Dropdown(
        id='host_id_verified',

        options=[
            {'label': 'Yes', 'value': '1'},
            {'label': 'No', 'value': '0'},],
        value='1')]),

    html.Br(),

    html.Div(["Is the AirBnB instantly bookable?: "]),
    dcc.Dropdown(
        id='instant_bookable',

        options=[
            {'label': 'Yes', 'value': '1'},
            {'label': 'No', 'value': '0'},],
        value='1'),

    html.Br(),

    html.Div(["Number Of Reviews: ",
              dcc.Input(id='number_of_reviews', value='1', type='text')]),
    html.Br(),

    html.Div(["Number of Bedrooms: ",
              dcc.Input(id='bedrooms', value='1.0', type='text')]),
    html.Br(),

    html.Div(["Number of Beds: ",
              dcc.Input(id='beds', value='1.0', type='text')]),
    html.Br(),

    html.Div([
        html.Button('Submit', id='submit',n_clicks=0),

    html.Br(),
    html.Br(),

    html.Div(id = 'my-output'),

    html.Br(),

    html.Div('Refresh before new prediction!'),
        
    ])

    ]

    )

@app.callback(
    [Output('my-output','children')],
    [Input('my-input','value'),
    Input('city-dropdown','value'),
    Input('property-type-dropdown','value'),
    Input('room_type_dropdown','value'),
    Input('accommodates_input','value'),
    Input('bathrooms_input','value'),
    Input('bed_types_dropdown','value'),
    Input('cancellation_policy_dropdown','value'),
    Input('cleaning_fee_check','value'),
    Input('profile_pic_check','value'),
    Input('host_id_verified','value'),
    Input('instant_bookable','value'),
    Input('longitude_input','value'),
    Input('number_of_reviews','value'),
    Input('bedrooms','value'),
    Input('beds','value'),
    Input('submit','n_clicks'),])


def update_output_div(latitude, city, property_type,room_type,
                    accommodates_value, bathrooms_value, bed_type_value,
                    cancellation_policy, cleaning_fee, profile_pic_check,
                    host_id, instant_bookable, longitude_input,
                    number_of_reviews, bedrooms, beds,n_clicks):

    if(n_clicks == 1):
        df_for_prediction = get_inputs(property_type, room_type, int(accommodates_value), float(bathrooms_value), 
        bed_type_value, cancellation_policy, bool(cleaning_fee), city, int(profile_pic_check), 
         int(host_id), int(instant_bookable), float(latitude), float(longitude_input), 
         int(number_of_reviews), float(bedrooms), float(beds), X_train, X_test)


        prediction = result.predict([df_for_prediction])[len(result.predict([df_for_prediction]))-1]
        number = float(((prediction[0])[0]))
        return ['Predicted Price of AirBnB: ${}'.format(round(number, 2))]
    else:
        return [['Predicted Price of AirBnB:']]



layout = dbc.Row([column1, column2])
