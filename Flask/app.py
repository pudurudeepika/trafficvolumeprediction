import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = joblib.load(open('D:\Traffic_Volume_Project\model.pkl', 'rb'))
scale = joblib.load(open('D:\Traffic_Volume_Project\scale.pkl', 'rb'))
@app.route('/')  # route to display the home page
def home():
    return render_template('index.html')  # rendering the home page

@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def predict():
    # reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    features_values = np.array(input_feature).reshape(1, -1)
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']
    data = pd.DataFrame(features_values, columns=names)
    data = scale.transform(data)
    data = pd.DataFrame(data, columns=names)
    # predictions using the loaded model file
    prediction = model.predict(data)
    # print(prediction)
    text = "Estimated Traffic Volume is :" + str(prediction[0])
    return render_template("index.html", prediction_text=text)
    # return render_template("result.html", prediction_text=text, url=data:image/png;base64,{{ url }})

# showing the prediction results in a UI
if __name__ == "__main__":
    app.run(debug=True)


