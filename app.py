import pandas as pd
import numpy as np
import scikit_learn as skl 

from scikit_learn.model_selection import train_test_split
from scikit_learn.preprocessing import MinMaxScaler, StandardScaler
from scikit_learn.decomposition import PCA
from scikit_learn.metrics import mean_squared_error, mean_absolute_error

# -----------------------
# Load and prepare dataset
# -----------------------
df = pd.read_csv('dielectron.csv')
df = df.dropna()

target_col = 'pt1'

features = ['E1', 'px1', 'py1', 'pz1', 'eta1', 'phi1',
            'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2',
            'Q1', 'Q2']

X = df[features].values
y = df[[target_col]].values  
X = StandardScaler().fit_transform(X)

y_scaler = MinMaxScaler((0, 1))
y = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)