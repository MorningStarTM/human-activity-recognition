from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.svm import SVC
import pickle
import time

app = Flask(__name__)

class_names = ['downstairs', 'jogging', 'sitting', "standing", "upstairs", "walking", "Activity_1",
                  "Activity_2", "Activity_3"]

def sample_dict(df1):
    d = {'acc_x_mean':[df1['acc_x'].mean()],
        'acc_y_mean':[df1['acc_y'].mean()],
        'acc_z_mean':[df1['acc_z'].mean()],
        'acc_x_std':[df1['acc_x'].std()],
        'acc_y_std':[df1['acc_y'].std()],
        'acc_z_std':[df1['acc_z'].std()],
        'acc_x_max':[df1['acc_x'].max()],
        'acc_y_max':[df1['acc_y'].max()],
        'acc_z_max':[df1['acc_z'].max()],
        'acc_x_min':[df1['acc_x'].min()],
        'acc_y_min':[df1['acc_y'].min()],
        'acc_z_min':[df1['acc_z'].min()],
        'gyro_x_mean':[df1['gyro_x'].mean()],
        'gyro_y_mean':[df1['gyro_y'].mean()],
        'gyro_z_mean':[df1['gyro_z'].mean()],
        'gyro_x_std':[df1['gyro_x'].std()],
        'gyro_y_std':[df1['gyro_y'].std()],
        'gyro_z_std':[df1['gyro_z'].std()],
        'gyro_x_max':[df1['gyro_x'].max()],
        'gyro_y_max':[df1['gyro_y'].max()],
        'gyro_z_max':[df1['gyro_z'].max()],
        'gyro_x_min':[df1['gyro_x'].min()],
        'gyro_y_min':[df1['gyro_y'].min()],
        'gyro_z_min':[df1['gyro_z'].min()],
                   }
    return d



@app.route('/')
def home():
    return 'Welcome to CADS'

@app.route('/prediction')
def predict():
    data = request.get_json()
    #min_length = min(len(data[key]) for key in data)
    #filtered_data = {key: values for key, values in data.items() if len(values) >= min_length}
    df = pd.DataFrame(data, index=False)
    test_data = sample_dict(df)
    df = pd.DataFrame(test_data, index=False)
    file = open("E:\\github_clone\\human-activity-recognition\\models\\SVC_model.pkl", 'rb')
    model = pickle.load(file)
    time.sleep(5)
    prediction = model.predict(test_data)[0]
    print(class_names[prediction])
    return prediction, class_names[prediction]

if __name__ == "__main__":
    app.run(debug=True)