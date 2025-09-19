import numpy as np
import os
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    sepal_length = float(request.form['SepalLength'])
    sepal_width  = float(request.form['SepalWidth'])
    petal_length = float(request.form['PetalLength'])
    petal_width  = float(request.form['PetalWidth'])

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    return render_template('result.html', prediction=prediction[0])
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT env variable
    app.run(host="0.0.0.0", port=port)
