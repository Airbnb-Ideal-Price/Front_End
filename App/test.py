from tensorflow.keras.models import load_model
import tensorflow.keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

array2 = ('Apartment', 'Entire home/apt', 3, 1.0, 'Real Bed',
         'strict', True, 'NYC', 1, 
         1, 0, 40.696524, -73.991617, 
         2, 1.0, 1.0)
        
df = pd.read_csv('train.csv')

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

def prepare_inputs(X_train, X_test):
    X_train_enc, X_test_enc = list(), list()
    # label encode each column
    for i in range(X_train.shape[1]):
        le = LabelEncoder()
        # encode
        train_enc = le.fit_transform(X_train.iloc[:, i].values)
        test_enc = le.fit_transform(X_test.iloc[:, i].values)
        # store
        X_train_enc.append(train_enc)
        X_test_enc.append(test_enc)
    return X_train_enc, X_test_enc

def scale_inputs(X_train, X_test):
    ss = StandardScaler()
    ss.fit_transform(X_train, X_test)
    return X_train, X_test

def get_inputs(property_type, room_type, accomodates, bathrooms, bed_type,
         cancellation_policy, cleaning_fee, city, host_has_profile_pic, 
         host_identity_verified, instant_bookable, latitude, longitude, 
         number_of_reviews, bedrooms, beds, X_train, X_test):
         
    values_for_prediction = (property_type, room_type, accomodates, bathrooms, bed_type,
         cancellation_policy, cleaning_fee, city, host_has_profile_pic, 
         host_identity_verified, instant_bookable, latitude, longitude, 
         number_of_reviews, bedrooms, beds)



    df_for_prediction = pd.DataFrame(values_for_prediction).T
    df_for_prediction.columns = X.columns

    X_test = X_test.append(df_for_prediction)

    X_train, X_test = prepare_inputs(X_train, X_test)

    X_train, X_test = scale_inputs(X_train, X_test)

    return(X_test)
