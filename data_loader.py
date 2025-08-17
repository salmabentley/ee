import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(file_path="dielectron.csv", target_col="pt1", test_size=0.3):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    features = [
        "E1","px1","py1","pz1","eta1","phi1",
        "E2","px2","py2","pz2","pt2","eta2","phi2",
        "Q1","Q2"
    ]
    X = df[features].values
    y = df[[target_col]].values

    X = StandardScaler().fit_transform(X)
    y_scaler = MinMaxScaler((0,1))
    y = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test, y_scaler
