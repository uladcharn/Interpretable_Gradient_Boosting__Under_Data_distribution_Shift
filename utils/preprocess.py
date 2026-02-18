import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd # load and manipulate data and for One-Hot Encoding
import numpy as np # calculate the mean and standard deviation
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder, OneHotEncoder

from sklearn.preprocessing import KBinsDiscretizer

def preliminary_preproc(dir):
    """Initial preprocessing. Add any general preprocessing steps to this function."""
    df = pd.read_csv(dir)

    df.drop('CustomerID', axis=1, inplace=True) # uninformative column
    df.drop('Count', axis=1, inplace=True) # uninformative column
    df.drop('Country', axis=1, inplace=True) # only one country is present - United States
    df.drop('State', axis=1, inplace=True)  # only one state is present - California
    df.drop('Lat Long', axis=1, inplace=True) # uninformative column
    df.drop('Zip Code', axis=1, inplace=True) # uninformative column - we discriminate by cities; no additional info provided about zipcodes
    df.drop('Churn Score',axis=1, inplace=True) # removing a Churn Score proxy to prevent target leakage 
    df.drop('Churn Reason',axis=1, inplace=True) # removing a Churn Label proxy to prevent target leakage 
    df.drop('Churn Label', axis=1, inplace=True) # we have Churn Value already
    
    df['Total Charges'] = pd.to_numeric(df['Total Charges'].str.strip(), errors='coerce').fillna(0)

    ## NOTE: No missing values detected among categorical variables

    # creating Dependent and Independent variables

    X = df.drop('Churn Value', axis=1).copy()
    y = df['Churn Value']

    return X, y, df

def separate_data_shift_categories(df):
    df_collection = {'Tenure': {}, 'Status': {}, 'City':{}}

    # Tenure

    df_collection['Tenure']['Short'] = df[df['Tenure Months'] <= 12].copy()
    df_collection['Tenure']['Medium'] = df[(df['Tenure Months'] > 12) & (df['Tenure Months'] <= 60)].copy()
    df_collection['Tenure']['Long'] = df[df['Tenure Months'] > 60].copy()

    # City

    top_3_city = df['City'].value_counts().head(3).index.tolist()
    top_3_15_city = df['City'].value_counts().iloc[3:15].index.tolist()
    top_15_and_above_city = df['City'].value_counts().iloc[15:].index.tolist()

    df_collection['City']['Top_3'] = df[df['City'].isin(top_3_city)].copy()
    df_collection['City']['Top_3_to_15'] = df[df['City'].isin(top_3_15_city)].copy()
    df_collection['City']['Top_15_and_above'] = df[df['City'].isin(top_15_and_above_city)].copy()

    # Family Status

    from itertools import product
    genders = df['Gender'].unique()
    partner = df['Partner'].unique()
    dependents = df['Dependents'].unique()
    status_categories = list(product(genders,partner,dependents))

    for g, p, d in status_categories:
        # Use a descriptive key for the dictionary
        key = f"{g}_{p}_{d}"
        df_collection['Status'][key] = df[(df['Gender'] == g) & (df['Partner'] == p) &
                                            (df['Dependents'] == d)].copy()

    return df_collection

def preprocess_shift_categories_da(df_sc, test_cat = ['Short'], blacklist = ['Short', 'Medium', 'Long', 'Total Charges', 'Tenure Months']):
    """ 
    Docstring for preprocess_shift_categories
    
    :param df_sc: dataset for a selected shift category
    :param test_cat: a list of test categories
    :param split_criteria: split criteria (Tenure Months by default)
    :param blacklist: a list of features to be excluded 
    """

    all_features = [col for col in list(df_sc[test_cat[0]].columns) if col not in ['Churn Value'] + blacklist]

    # Numbers vs Strings
    numeric_features = df_sc[test_cat[0]][all_features].select_dtypes(include=['number']).columns.tolist()
    obj_features = df_sc[test_cat[0]][all_features].select_dtypes(exclude=['number']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        # Scales all numeric behavioral data (MonthlyCharges, Tenure, etc.)
        ('numeric', StandardScaler(), numeric_features),
        
        # Encodes all categorical behavioral data (PaymentMethod, Contract, etc.)
        ('object', TargetEncoder(handle_unknown='value'), obj_features)
        
    ], remainder='drop') # Drops the Split Criteria columns automatically

    train_names = [name for name in df_sc.keys() if name not in test_cat]

    X_train = pd.concat([df_sc[name].drop('Churn Value', axis=1).copy() for name in train_names], axis = 0, ignore_index=True)
    y_train = pd.concat([df_sc[name]['Churn Value'].copy() for name in train_names], axis = 0, ignore_index=True)
    X_test = pd.concat([df_sc[name].drop('Churn Value', axis=1).copy() for name in test_cat] , axis = 0, ignore_index=True)
    y_test = pd.concat([df_sc[name]['Churn Value'].copy() for name in test_cat], axis = 0, ignore_index=True)

    X_train_preproc = preprocessor.fit_transform(X_train, y_train)
    X_test_preproc = preprocessor.transform(X_test)

    # Convert back to DataFrame for easier SHAP analysis later
    # (TargetEncoder/StandardScaler usually return numpy arrays)
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_preproc, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_preproc, columns=feature_names, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test, numeric_features, obj_features