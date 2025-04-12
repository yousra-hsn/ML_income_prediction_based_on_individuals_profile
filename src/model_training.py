"""
model.py : Contient des fonctions pour entraîner différents modèles de machine learning et les sauvegarder.
Fonctions : train_random_forest(), train_logistic_regression(), train_svm(), save_model(), load_model()
"""

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
#from pipelines import get_logistic_regression_pipeline, get_random_forest_pipeline

def train_predict_model_adaboost(pipeline, df_train, df_test, model_name):
    
    y_train = df_train['>50K'].values
    y_test = df_test['>50K'].values

    # Entraînement du modèle
    pipeline_fitted = pipeline.fit(df_train, y_train)
    
    # Obtenir les données transformées avant l'entraînement
    test = pipeline.named_steps['preprocessor']
    print(test.shape())
    X_train_transformed = test.transform(df_train.drop(columns=['>50K']))
    feature_names = test.get_feature_names_out()
    df_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    
    
    # Prédictions sur les données de validation
    y_pred = pipeline_fitted.predict(df_test)
    
    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy}")
    
    # Sauvegarde du modèle entraîné
    joblib.dump(pipeline_fitted, f'../models/{model_name}.pkl')
    
    return pipeline_fitted, y_test, y_pred, df_train_transformed

def best_param_grid_search(model, X_train_up,y_train_up, X_train, y_train, model_name): 
    param_grid = {'C': [0.001, 0.01, 0.1, 1., 10.], 'max_iter': [100, 200, 300]}
    grid = GridSearchCV(model, param_grid, verbose=False, n_jobs=1, return_train_score=True, scoring='f1')
    grid.fit(X_train_up, y_train_up)
    print("meilleur parametres : ",grid.best_params_)
    grid_model= grid.best_estimator_
    grid_model_fitted = grid_model.fit(X_train, y_train)
    # Sauvegardez le modèle dans un fichier apres entrainement
    joblib.dump(grid_model_fitted, '../models/'+model_name+'_fitted.pkl')
    
    return grid_model,grid_model_fitted