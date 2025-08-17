import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(file_path="dielectron.csv", target_col="pt1", test_size=0.3):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print("pt1 min/max in CSV:", df[target_col].min(), df[target_col].max())

    # Shuffle the dataframe to ensure random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    features = [
        "E1","px1","py1","pz1","eta1","phi1",
        "E2","px2","py2","pz2","pt2","eta2","phi2",
        "Q1","Q2"
    ]
    X = df[features].values
    y = df[[target_col]].values

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print("y_train min/max:", y_train.min(), y_train.max())
    print("y_test min/max:", y_test.min(), y_test.max())
    return X_train, X_test, y_train, y_test
