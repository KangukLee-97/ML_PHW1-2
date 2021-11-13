import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def findBest(X, y, scaled_col, encoded_col, scalers, encoders, models, model_param, cv):

# Open the datasets (breast-cancer-wisconsin)
# Apply the column names
df = pd.read_csv("breast-cancer-wisconsin.data",
                 names=["Sample code number", "Clump Thickness",
                        "Uniformity of Cell Size", "Uniformity of Cell Shape",
                        "Marginal Adhesion", "Single Epithelial Cell Size",
                        "Bare Nuclei", "Bland Chromatin",
                        "Normal Nucleoli", "Mitoses", "Class"])

# Find the missing values and remove (Bare Nuclei = ? value)
# Preprocessing step
df = df.replace('?', np.NaN)   # Change the ? value to NaN
df = df.dropna(axis=0)   # Drop the rows that include NaN
df = df.drop("Sample code number", axis=1)   # Drop the sample code number because it doesn't need

# Set the feature and target value
X = df.drop(['Class'], axis=1)   # Class를 제외한 feature
y = df['Class']   # target value (Class)

# Define the tuple list for each model, scaler and encoder
models, scalers, encoders = [], [], []
models.append(
    ("LR", LogisticRegression()),
    ("DTE", DecisionTreeClassifier()),
    ("DTG", DecisionTreeClassifier()),
    ("SVM", SVC())
)
scalers.append(
    ("SS", StandardScaler()),
    ("MMS", MinMaxScaler()),
    ("RBS", RobustScaler()),
    ("MAS", MaxAbsScaler())
)
encoders.append(
    ("Ord", OrdinalEncoder()),
    ("Hot", OneHotEncoder()),
    ("Lbl", LabelEncoder())
)