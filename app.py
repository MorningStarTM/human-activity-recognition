from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

class_names = ['downstairs', 'jogging', 'sitting', "standing", "upstairs", "walking", "Activity_1",
                  "Activity_2", "Activity_3"]


@app.route('/')
def home():
    return 'Welcome to CADS'

@app.route('/prediction')
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=False)
    file = open("E:\\github_clone\\human-activity-recognition\\models\\SVC_model.pkl", 'rb')
    model = pickle.load(file)
    prediction = model.predict(df)[0]
    print(class_names[prediction])
    return prediction, class_names[prediction]

if __name__ == "__main__":
    app.run(debug=True)