import sys
import os

# Add the project directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the required modules from lib_import.py
from lib.lib_import import fetch_ucirepo, pd

def load_data():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 

    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 

    data = pd.concat([X, y], axis=1)
    return data