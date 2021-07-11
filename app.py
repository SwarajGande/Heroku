import numpy as np
from flask import Flask,render_template, request
import pickle

# Intializing the WSGI
app = Flask(__name__)

# Opening the Pickle File
model = pickle.load(open("my_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods =["POST"])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    predict = model.predict(final_features)
    output = round(predict[0],2)

    return render_template("index.html",prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug =True)
