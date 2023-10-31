from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to CADS'

@app.route('/prediction')
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=False)
    file = open("E:\\github_clone\\human-activity-recognition\\models\\SVC_model.pkl", 'rb')
    model = pickle.load(file)
    prediction = model.predict(df)
    print(prediction)
    return prediction

if __name__ == "__main__":
    app.run(debug=True)