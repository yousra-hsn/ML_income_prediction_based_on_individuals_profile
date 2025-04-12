
"""
model_evaluation.py : Contient des fonctions pour évaluer les performances des modèles.
Fonctions : plot_confusion_matrix_sns(), save_image()
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from yellowbrick.model_selection import LearningCurve


def plot_confusion_matrix_sns(y_test, y_pred, model_name):
    """
    Affiche la matrice de confusion pour les prédictions d'un modèle.
    """
    """
    Affiche la matrice de confusion pour les prédictions d'un modèle.
    """

    print('='*30)
    print(model_name)
    print('='*30, '\n')

    print("Matrice de confusion:")
    c_matrix = confusion_matrix(y_test, y_pred)
    print(c_matrix, '\n') # afficher à l'écran notre matrice de confusion
    print("Rapport de classification:")
    print(classification_report(y_test, y_pred), '\n')
    print('Exactitude: %f' %(accuracy_score(y_test, y_pred)*100), '\n')

    # Affichage et enregistrement de la matrice de confusion avec Matplotlib
    plt.figure(figsize=(8, 4))
    plt.imshow(c_matrix, interpolation='nearest', cmap='Blues')
    plt.title(f'{model_name}')
    plt.colorbar()

    tick_marks = np.arange(len(set(y_test)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Ajouter les annotations
    thresh = c_matrix.max() / 2.
    for i, j in np.ndindex(c_matrix.shape):
        plt.text(j, i, format(c_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if c_matrix[i, j] > thresh else "black")

    save_image(plt, model_name)

   
def save_image(plt,filename):
    # Ajuster les marges pour que le titre soit complètement visible
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Enregistrer le schéma dans le dossier 'images'
    plot_filename = '../images/' + filename + '.png'
    plt.savefig(plot_filename)  # Enregistre le schéma
    plt.show()  # Affiche le schéma
    plt.close()  # Ferme la figure pour éviter les conflits de figure
    

def print_score_validation(model, X_train_up, y_train_up, list_hyperparams, model_name):
    list_hyperparams = np.linspace(0.0001, 0.09, 30)
    train_score, val_score = validation_curve(model,
                                          X_train_up,
                                          y_train_up,
                                          param_name='C', 
                                          param_range=list_hyperparams, 
                                          cv=5,
                                         scoring="f1",)

    plt.figure(figsize=(12, 4))
    plt.plot(list_hyperparams, train_score.mean(axis = 1), label = 'train')
    plt.plot(list_hyperparams, val_score.mean(axis = 1), label = 'validation')
    plt.legend()
    plt.title("Courbe de validation pour " + model_name)
    plt.ylabel('score')
    plt.xlabel('Paramètre de régularisation: ' r'$\lambda = \frac{1}{C}$')
    plt.show()
    


def print_courbe_apprentissage(model, X_train_up, y_train_up, model_name,score): 
    N, train_score, val_score = learning_curve(model, X_train_up, y_train_up, 
                                           cv=5, scoring=score,
                                           train_sizes=np.linspace(0.1, 1, 10))

    plt.figure(figsize=(12,8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.title("Courbe d'apprentissage pour "+ model_name)
    