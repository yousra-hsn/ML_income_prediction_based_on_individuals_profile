import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_installation(package):
    try:
        __import__(package)
        print(f"{package} installed successfully.")
    except ImportError:
        print(f"Failed to install {package}.")

packages = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "plotly",
    "ucimlrepo",
    "missingno",
    "imblearn",
    "joblib",
    "yellowbrick",
    "xgboost",
    'optuna'
]

for package in packages:
    install(package)
    check_installation(package)
    
    