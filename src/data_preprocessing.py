import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def preprocess(df):
    df = fix_target(df)
    df = remove_inutile_column(df)
    df = drop_outliers(df)
    df = regroupe_categories(df)
    df = df.drop_duplicates()
    return df

def fix_target(data) :
    """
    Corrige la variable cible 'income' pour qu'elle prenne des valeurs binaires.
    
    Args:
        df (pd.DataFrame): DataFrame initiale contenant la variable cible 'income'.

    Returns:
        pd.DataFrame: DataFrame avec la variable cible corrigée.
    """
    data['income'] = data['income'].str.replace('<=50K.', '<=50K')
    data['income'] = data['income'].str.replace('>50K.', '>50K')
    data['>50K'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    data = data.drop(columns=['income'])
    return data

def remove_inutile_column(df) :
    df = df.drop("education", axis=1)
    return df

def impute_missing_cat_values(df_train,df_test,cat_features, strategy):
    imput_cat = SimpleImputer(missing_values=np.nan, strategy=strategy)
    for feature in cat_features :
        df_train[feature] = imput_cat.fit_transform(df_train[feature].values.reshape(-1,1)).ravel()
        df_test[feature] = imput_cat.transform(df_test[feature].values.reshape(-1,1)).ravel()
    return df_train, df_test



def standardize(df_train,df_test, cont_features):
    scaleStd = StandardScaler()
    
    for feature in cont_features:
        # Normaliser les données d'entrainement
        df_train[feature] = scaleStd.fit_transform(df_train[feature].values.reshape(-1,1))
        df_train[feature] = df_train[feature].ravel()

        # Normaliser les données test
        df_test[feature] = scaleStd.transform(df_test[feature].values.reshape(-1,1))
        df_test[feature] = df_test[feature].ravel()
        
    return df_train, df_test

def seperate_train_test(df,random_state):
    """
    Sépare le DataFrame en un ensemble d'entraînement et un ensemble de test.
    
    Args:
        df (pd.DataFrame): Le DataFrame à diviser.
        random_state (int): La graine aléatoire pour la reproductibilité.
    
    Returns:
        tuple: Un tuple contenant les ensembles d'entraînement et de test.
    """
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df['>50K'])
    return df_train, df_test


def transform_to_float(df):
    df[['age', 'capital-gain', 'capital-loss', 'hours-per-week','fnlwgt']] = df[['age', 'capital-gain', 
                                                                                 'capital-loss', 'hours-per-week',
                                                                                 'fnlwgt']].astype(float)
    return df

def get_column_min_max(df, column_name):
    """
    Renvoie les valeurs minimum et maximum d'une colonne spécifique dans un DataFrame.
    
    Parameters:
        df (pd.DataFrame): Le DataFrame pandas.
        column_name (str): Le nom de la colonne dont on veut obtenir les valeurs min et max.
    
    Returns:
        tuple: Un tuple contenant la valeur minimum et la valeur maximum de la colonne.
    """
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    return min_value, max_value


def get_cont_features(df):
    cont_features = df.select_dtypes('int64').columns
    ## supprimer 50K
    cont_features = cont_features.drop('>50K')
    return cont_features

def get_cat_features(df):
    cat_features = df.select_dtypes('object').columns
    # ajout de education-num dans cat_features
    cat_features = cat_features.append(pd.Index(['education-num']))
    return cat_features


def encode_cat_features(df_train,df_test, cat_features):
    encoder = LabelEncoder()
    for feature in cat_features:
        df_train[feature] = encoder.fit_transform(df_train[feature])
        df_test[feature] = encoder.transform(df_test[feature])
    return df_train, df_test


def encode_cat_features_onehot(df_train, df_test, cat_features):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit the encoder on the training data and transform both train and test data
    df_train_encoded = encoder.fit_transform(df_train[cat_features])
    df_test_encoded = encoder.transform(df_test[cat_features])
    
    # Get the feature names after encoding
    encoded_feature_names = encoder.get_feature_names_out(cat_features)
    
    # Convert the numpy arrays returned by the encoder to dataframes
    df_train_encoded = pd.DataFrame(df_train_encoded, columns=encoded_feature_names)
    df_test_encoded = pd.DataFrame(df_test_encoded, columns=encoded_feature_names)
    
    # Drop the original categorical columns and concatenate the new one-hot encoded columns
    df_train = df_train.drop(cat_features, axis=1)
    df_test = df_test.drop(cat_features, axis=1)
    
    df_train = pd.concat([df_train.reset_index(drop=True), df_train_encoded.reset_index(drop=True)], axis=1)
    df_test = pd.concat([df_test.reset_index(drop=True), df_test_encoded.reset_index(drop=True)], axis=1)
    
    return df_train, df_test


def drop_outliers(df):
    """Remarque : la suppression des outliers doit etre faite après la division des données en train et test et avant la normalisation des données.
    """
    df = df[df['capital-gain'] < 40000.]
    df = df[df['capital-loss'] < 4000.]

    return df


def regroupe_categories(df):
    df['workclass'] = df['workclass'].replace({'?': 'Not referenced'})
    df['native-country'] = df['native-country'].replace({'?': 'Not referenced'})
    df['occupation'] = df['occupation'].replace({'?': 'Not referenced'})
    # Regroup 'Without-pay' and 'Never-worked' to 'No revenu'
    df['workclass'] = df['workclass'].replace({'Without-pay': 'No revenu', 'Never-worked': 'No revenu'})

    # Regrouper 'Self-emp-not-inc' and 'Self-emp-inc' to 'Self-emp'
    df['workclass'] = df['workclass'].replace({'Self-emp-not-inc': 'Self-emp', 'Self-emp-inc': 'Self-emp'})
        
    df['marital-status'] = df['marital-status'].replace({'Divorced': 'Now Single', 'Separated': 'Now Single', 'Widowed': 'Now Single'})

    df['marital-status'] = df['marital-status'].replace({'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married'})
     
    df['relationship'] = df['relationship'].replace({'Husband': 'Married', 'Wife': 'Married'})
     
    df['race'] = df['race'].replace({'Amer-Indian-Eskimo': 'Other'})
   
    # Counting for each value of the variable 'native-country' the number of individuals who have this value
    filtered = df[df['native-country'] != 'United-States']

    for country in filtered['native-country'].unique():
        nb_samples_associated = filtered[filtered['native-country'] == country].shape[0]
        if nb_samples_associated < 200:
            df['native-country'] = df['native-country'].replace({country: 'Other'})
    
            
    return df
        

def getX_y(df):
    X = df.drop(columns=['>50K'])
    y = df['>50K']
    return X, y


def selectKbest(df_train, target_train, df_test, k=10):
    """_summary_

    Args:
        df_train (dataframe): contenant que les features (sans la target)
        target_train (_type_): _description_
        df_test (dataframe): contenant que les features (sans la target)
        k (int, optional): _description_. Defaults to 10.

    Returns:
        df_train_select: dataframe contenant les k features les plus importantes
        df_test_select : dataframe contenant les k features les plus importantes
        new_feature_names : liste des noms des k features les plus importantes
    """
    
    select = SelectKBest(score_func=chi2, k=k)
    select.fit_transform(df_train, target_train)
    mask = select.get_support(indices=True)
    new_feature_names = df_train.columns[mask]
    # Reconstruire les ensembles train et test
    df_train_select = df_train[new_feature_names]
    df_test_select = df_test[new_feature_names]
    return df_train_select, df_test_select, new_feature_names