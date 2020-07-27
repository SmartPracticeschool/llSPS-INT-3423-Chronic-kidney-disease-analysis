import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
app = Flask(__name__)
model = pickle.load(open('DT.pkl', 'rb'))
l=['NOT GET CHRONIC KIDNEY DISEASE','GET CHRONIC KIDNEY DISEASE']
@app.route('/')
def home():
    return render_template('Chronic_kidney.html')

@app.route('/y_pred',methods=['POST'])
def y_pred():
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
    sc=load('DT.save')
    prediction = model.predict(sc.fit_transform(x_test))
    output=prediction[0]
    return render_template('Chronic_kidney.html', prediction_text='THIS PERSON MAY {}'.format(l[output]))

if __name__ == "__main__":
    app.run(debug=True)
