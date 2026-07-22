import joblib

from sklearn.preprocessing import StandardScaler


def fit_scaler(X):

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, scaler


def transform_scaler(X, scaler):

    return scaler.transform(X)


def save_scaler(scaler, path):

    joblib.dump(scaler, path)


def load_scaler(path):

    return joblib.load(path)