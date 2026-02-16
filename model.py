import joblib

def load_model(path="churn_model.pkl"):
    return joblib.load(path)

def predict_churn(model, X):
    probs = model.predict(X)[:, 1]
    return probs